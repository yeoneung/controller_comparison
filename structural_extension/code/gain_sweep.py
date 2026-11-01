"""Year-long targets and repeated controller comparisons with shared input tapes."""
from pathlib import Path
import argparse,ctypes,json,math,platform,subprocess,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CONFIG={'horizon':105120,'dt_seconds':300.,'comparisons':256,'gain_min':.6,'gain_max':1.8,
 'increment':.01,'paths':64,'timing_repetitions':3,'validation_paths':8,
 'validation_gain_indices':[0,63,127,191,255],
 'conditions':[{'zones':6,'wall_capacity':8e6},{'zones':18,'wall_capacity':16e6}],
 'bias_budget_K':.001,'maximum_memory_search':10000,'seed':2026103191,
 'air_capacity':2e6,'air_wall':150.,'wall_outdoor':100.,'interzone':50.,'reference_power':1000.,
 'scope':'Illustrative physical model; fixed truncation-bias budget. No building calibration or sampling-error certificate.'}

def setup(wall_capacity):
    p=CONFIG;dt=p['dt_seconds'];ca=p['air_capacity'];ga=p['air_wall'];go=p['wall_outdoor']
    M=np.array([[1-dt*ga/ca,dt*ga/ca],[dt*ga/wall_capacity,1-dt*(ga+go)/wall_capacity]])
    b=np.array([dt*p['reference_power']/ca,0.]);period=288;t=np.arange(period);phase=2*np.pi*t/period
    moment=np.column_stack((dt/ca*np.hypot(400+100*np.sin(phase-np.pi/2),150),
        dt*go/wall_capacity*np.hypot(6+2*np.sin(phase),1.5)))
    cycle=np.zeros(2)
    for u in moment:cycle=M@cycle+u
    r0=np.linalg.solve(np.eye(2)-np.linalg.matrix_power(M,period),cycle)
    periodic=np.empty((period+1,2));periodic[0]=r0
    for t in range(period):periodic[t+1]=M@periodic[t]+moment[t]
    np.testing.assert_allclose(periodic[-1],r0,atol=1e-12)
    cap=p['maximum_memory_search'];P=np.empty((cap+1,2,2));P[0]=np.eye(2)
    S=np.zeros((cap+1,2))
    for t in range(cap):P[t+1]=M@P[t];S[t+1]=M@S[t]+b
    Sinf=np.linalg.solve(np.eye(2)-M,b)
    assert periodic.min()>0 and max(abs(np.linalg.eigvals(M)))<1
    return M,b,periodic,P,S,Sinf

def select(wall_capacity,exhaustive=False):
    M,b,r,P,S,Sinf=setup(wall_capacity);cfg=CONFIG
    H=cfg['horizon'];d=cfg['increment'];target=cfg['bias_budget_K']/d
    amin=cfg['gain_min'];amax=cfg['gain_max']+d;lip=4/(3*math.sqrt(3))
    def evaluate(L):
        prefix=r[(H-L)%288];state=float((P[L]@prefix)[0]);initial=float((P[L]@Sinf)[0])
        if L:
            h=P[:L]@prefix
            f=b[0]*(np.minimum(h[:,0],2)+amax*S[:L,0]*np.minimum(lip*h[:,0],1))
            separate=initial+float(P[L-1::-1,0,0]@f)
        else:separate=Sinf[0]
        generic=min(separate,2*state/d,Sinf[0]+S[L,0])
        structural=initial+(1/amin+2*S[L,0])*state
        return generic,min(generic,structural)
    def bisect(column):
        lo=0;hi=cfg['maximum_memory_search'];assert evaluate(hi)[column]<=target
        if evaluate(0)[column]<=target:return 0
        while hi-lo>1:
            mid=(lo+hi)//2
            if evaluate(mid)[column]<=target:hi=mid
            else:lo=mid
        return hi
    old,new=bisect(0),bisect(1)
    assert new<=old and max(evaluate(old)[0],evaluate(new)[1])<=target
    rows=None
    if exhaustive:
        rows=np.array([evaluate(L) for L in range(cfg['maximum_memory_search']+1)])
        assert int(np.flatnonzero(rows[:,0]<=target)[0])==old
        assert int(np.flatnonzero(rows[:,1]<=target)[0])==new
    return {'old_memory':old,'new_memory':new,'old_bias_bound_K':d*evaluate(old)[0],
        'new_bias_bound_K':d*evaluate(new)[1],'old_bound_at_new_K':d*evaluate(new)[0],
        'horizon_over_old_memory':H/old,'horizon_over_new_memory':H/new,
        'memory_reduction':1-new/old,'selection_work_independent_of_horizon':True},rows

def compile_kernel():
    output=ROOT/'build'/('gain_sweep.dylib' if sys.platform=='darwin' else 'gain_sweep.so')
    output.parent.mkdir(exist_ok=True);start=time.perf_counter()
    cmd=['cc','-O3','-std=c99','-dynamiclib' if sys.platform=='darwin' else '-shared']
    if sys.platform!='darwin':cmd.append('-fPIC')
    subprocess.run(cmd+[str(Path(__file__).with_suffix('.c')),'-o',str(output),'-lm'],check=True,capture_output=True)
    elapsed=time.perf_counter()-start;f=ctypes.CDLL(str(output)).gain_sweep
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    f.argtypes=[ptr]*3+[ctypes.c_int]*4+[ctypes.c_double]*8+[ptr]*2;f.restype=ctypes.c_int
    return f,elapsed

def simulate(f,z,weather,load,zones,wall,gains):
    cfg=CONFIG;out=np.empty((len(z),len(gains)))
    assert f(z,weather,load,len(z),z.shape[1],zones,len(gains),cfg['dt_seconds'],cfg['air_capacity'],wall,
        cfg['air_wall'],cfg['wall_outdoor'],cfg['interzone'],cfg['reference_power'],cfg['increment'],gains,out)==0
    return out

def tapes(memory,zones,paths,seed):
    # Reverse-time generation gives exactly matching suffixes at every memory.
    z=np.empty((paths,memory,zones+2))
    for j in range(paths):
        rng=np.random.default_rng(np.random.SeedSequence([seed,j]))
        z[j]=rng.standard_normal((memory,zones+2))[::-1]
    t=np.arange(CONFIG['horizon']-memory,CONFIG['horizon']);phase=2*np.pi*t/288
    return z,np.ascontiguousarray(6+2*np.sin(phase)),np.ascontiguousarray(400+100*np.sin(phase-np.pi/2))

def independent_check(f):
    rng=np.random.default_rng(83);z=rng.normal(size=(2,11,8));gains=np.array([.6,1.2,1.8])
    weather=np.linspace(3,8,11);load=np.linspace(250,500,11);got=simulate(f,z,weather,load,6,8e6,gains)
    ref=np.zeros_like(got)
    for pi,path in enumerate(z):
        a=np.zeros((2,len(gains),6));w=a.copy()
        for t in range(11):
            heat=load[t]+150*(.5*path[t,1]+np.sqrt(.75)*path[t,2:]);outside=weather[t]+1.5*path[t,0]
            gains2=np.stack((gains,gains+.01))[:,:,None]
            an=a+300/2e6*(150*(w-a)+50*(np.roll(a,1,axis=2)+np.roll(a,-1,axis=2)-2*a)+heat-1000*gains2*np.tanh(a))
            w=w+300/8e6*(150*(a-w)+100*(outside-w));a=an
        ref[pi]=(a[1]-a[0]).mean(axis=1)
    np.testing.assert_allclose(got,ref,atol=2e-15,rtol=2e-10)
    x,_,_=tapes(11,6,2,1);y,_,_=tapes(7,6,2,1)
    np.testing.assert_array_equal(x[:,-7:],y)
    return float(np.max(abs(got-ref)))

def run():
    for name in ['evidence','recorded_data']:(ROOT/name).mkdir(exist_ok=True)
    (ROOT/'recorded_data/protocol.json').write_text(json.dumps(CONFIG,indent=2))
    f,compile_seconds=compile_kernel();error=independent_check(f)
    gains=np.linspace(CONFIG['gain_min'],CONFIG['gain_max'],CONFIG['comparisons']);raw={};results=[]
    for ci,condition in enumerate(CONFIG['conditions']):
        zones=condition['zones'];wall=condition['wall_capacity'];start=time.perf_counter()
        choice,_=select(wall);prep=time.perf_counter()-start
        _,curves=select(wall,exhaustive=True);raw[f'bounds_{ci}']=curves
        print(json.dumps({'condition':ci,**choice}),flush=True)
        outputs=[];times=[]
        for rep in range(CONFIG['timing_repetitions']):
            pair=[None,None];durations=[0.,0.]
            for method in ([0,1] if rep%2==0 else [1,0]):
                start=time.perf_counter();z,weather,load=tapes(choice[['old_memory','new_memory'][method]],zones,CONFIG['paths'],CONFIG['seed']+100*ci+rep)
                pair[method]=simulate(f,z,weather,load,zones,wall,gains)
                durations[method]=time.perf_counter()-start
                del z
            outputs.append(pair);times.append(durations)
            print(json.dumps({'condition':ci,'repetition':rep,'times_seconds':durations}),flush=True)
        raw[f'contrasts_{ci}']=np.array(outputs);raw[f'times_{ci}']=np.array(times)
        selected=gains[CONFIG['validation_gain_indices']];validation=[]
        for memory in [CONFIG['horizon'],choice['old_memory'],choice['new_memory']]:
            z,weather,load=tapes(memory,zones,CONFIG['validation_paths'],CONFIG['seed']+1000+ci)
            validation.append(simulate(f,z,weather,load,zones,wall,selected));del z
        validation=np.array(validation);raw[f'validation_{ci}']=validation
        diff=validation[2]-validation[0];med=np.median(times,axis=0)
        result={**condition,**choice,'comparisons':len(gains),'paths':CONFIG['paths'],
            'compile_seconds':compile_seconds,'bound_selection_seconds':prep,
            'median_old_seconds':float(med[0]),'median_new_seconds':float(med[1]),
            'median_sampling_time_ratio':float(med[1]/med[0]),
            'median_complete_cost_ratio':float((compile_seconds+prep+med[1])/(compile_seconds+prep+med[0])),
            'median_seconds_saved':float(med[0]-med[1]),
            'validation_rms_error_K':float(np.sqrt(np.mean(diff**2))),
            'maximum_absolute_validation_error_K':float(np.max(abs(diff)))}
        results.append(result);print(json.dumps(result),flush=True)
    np.savez_compressed(ROOT/'recorded_data/gain_sweep.npz',gains=gains,**raw)
    report={'config':CONFIG,'results':results,'independent_kernel_error':error,'platform':platform.platform(),
        'cpu':'Apple M4 Pro','python':sys.version,'numpy':np.__version__,
        'cost_scope':'Each workload shares input generation and heat calculations across all 256 pairs. RNG, nonlinear states, allocation, memory selection and one-time compilation included; imports, serialization and full-horizon validation excluded.',
        'scope':'Three descriptive paired timings, not a high-confidence superiority statement. Prefix processing for selection is horizon-independent. Physical coefficients are illustrative.'}
    (ROOT/'evidence/gain_sweep.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(verify(),indent=2))

def verify():
    report=json.loads((ROOT/'evidence/gain_sweep.json').read_text());raw=np.load(ROOT/'recorded_data/gain_sweep.npz')
    assert report['config']==CONFIG and raw['gains'].shape==(256,)
    for i,r in enumerate(report['results']):
        choice,curves=select(r['wall_capacity'],exhaustive=True)
        for k,v in choice.items():assert r[k]==v
        np.testing.assert_allclose(raw[f'bounds_{i}'],curves)
        assert raw[f'contrasts_{i}'].shape==(3,2,64,256)
        med=np.median(raw[f'times_{i}'],axis=0)
        np.testing.assert_allclose(med,[r['median_old_seconds'],r['median_new_seconds']])
        val=raw[f'validation_{i}'];np.testing.assert_allclose(np.sqrt(np.mean((val[2]-val[0])**2)),r['validation_rms_error_K'])
        assert r['new_bias_bound_K']<=CONFIG['bias_budget_K']
    return {'passed':True,'workloads':len(report['results']),'queries_per_workload':256,
        'paths_per_query':64,'timing_repetitions':3,'all_results_retained':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--verify',action='store_true');parser.add_argument('--check-kernel',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    elif args.verify:print(json.dumps(verify(),indent=2))
    elif args.check_kernel:
        f,elapsed=compile_kernel()
        print(json.dumps({'passed':True,'independent_kernel_error':independent_check(f),
            'matching_rng_suffixes':True,'compile_seconds':elapsed},indent=2))
    else:parser.print_help()
