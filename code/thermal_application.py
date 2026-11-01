"""Physical RC-network example: fixed accuracy budgets and all declared conditions.

The model parameters are illustrative design values, not measured building data.
Run --run to create new observations/timings; --verify rechecks distributed data.
The full raw controller contrasts and timing repetitions are retained.
"""
from pathlib import Path
import argparse, ctypes, hashlib, json, math, os, platform, subprocess, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'recorded_data'
EVIDENCE = ROOT / 'evidence'
PROTOCOL = {
    'dt_seconds': 300., 'horizon': 8640, 'air_capacity_J_per_K': 2e6,
    'wall_capacities_J_per_K': [8e6, 16e6], 'air_wall_W_per_K': 150.,
    'wall_outdoor_W_per_K': 100., 'interzone_W_per_K': 50.,
    'reference_power_W': 1000., 'temperature_scale_K': 1.,
    'zones': [6, 18], 'gains': [0.8, 1.2, 1.8], 'increments': [0.01, 0.1],
    'absolute_contrast_bias_budget_K': 0.001,
    'weather_mean_K': 6., 'weather_daily_amplitude_K': 2., 'weather_sd_K': 1.5,
    'heat_mean_W': 400., 'heat_daily_amplitude_W': 100., 'heat_sd_W': 150.,
    'shared_heat_variance_fraction': 0.25,
    'accuracy_paths': 512, 'timing_paths': 256, 'batch_paths': 32,
    'timing_repetitions': 5, 'accuracy_seed': 2026103101, 'timing_seed': 2026103102,
    'primary_output': 'Mean zone-air temperature contrast in kelvin.',
    'baselines': ['matrix-resolved separate Lipschitz with saturation-range caps',
                  'two state-forgetting bounds without gain cancellation',
                  'global finite-gain tangent cap'],
    'scope': 'Numerical discrete-time model; no building calibration or continuous-time discretization certificate.'
}


def plant(wall_capacity):
    p = PROTOCOL; dt = p['dt_seconds']; ca = p['air_capacity_J_per_K']
    ga = p['air_wall_W_per_K']; go = p['wall_outdoor_W_per_K']
    return np.array([[1-dt*ga/ca, dt*ga/ca],
                     [dt*ga/wall_capacity, 1-dt*(ga+go)/wall_capacity]])


def schedules():
    t = np.arange(PROTOCOL['horizon']); phase = 2*np.pi*t/288
    return 6+2*np.sin(phase), 400+100*np.sin(phase-np.pi/2)


def arrays(wall_capacity):
    p = PROTOCOL; H = p['horizon']; M = plant(wall_capacity)
    b = np.array([p['dt_seconds']*p['reference_power_W']/p['air_capacity_J_per_K'], 0.])
    weather, load = schedules()
    moment = np.column_stack((p['dt_seconds']/p['air_capacity_J_per_K']*np.hypot(load,150),
        p['dt_seconds']*p['wall_outdoor_W_per_K']/wall_capacity*np.hypot(weather,1.5)))
    powers = np.empty((H+1,2,2)); powers[0] = np.eye(2)
    S = np.zeros((H+1,2)); R = np.zeros_like(S)
    for t in range(H):
        powers[t+1] = M@powers[t]; S[t+1] = M@S[t]+b
        R[t+1] = M@R[t]+moment[t]
    return M,b,powers,S,R


def bound_vectors(wall_capacity, theta, increment):
    """Full integer search; matrix powers and prefix moments used on both sides."""
    H = PROTOCOL['horizon']; M,b,P,S,R = arrays(wall_capacity)
    values = np.zeros(H+1); structural = np.zeros_like(values)
    lipschitz = 4/(3*math.sqrt(3))
    for L in range(H+1):
        s = H-L; initial = float((P[L]@S[s])[0])
        state = float((P[L]@R[s])[0])
        structural[L] = initial+(1/theta+2*S[L,0])*state
        if L:
            h = P[:L]@R[s]
            forcing = b[0]*(np.minimum(h[:,0],2)
                +(theta+increment)*S[:L,0]*np.minimum(lipschitz*h[:,0],1))
            separate = initial+float(P[L-1::-1,0,0]@forcing)
        else:
            separate = S[H,0]
        values[L] = min(separate, 2*state/increment, S[H,0]+S[L,0])
    combined = np.minimum(values,structural)
    target = PROTOCOL['absolute_contrast_bias_budget_K']/increment
    old = int(np.flatnonzero(values <= target)[0])
    new = int(np.flatnonzero(combined <= target)[0])
    assert old <= H and new <= old and values[H] == structural[H] == 0
    assert np.all(values[:old]>target) and np.all(combined[:new]>target)
    return {'generic_memory':old, 'combined_memory':new,
        'step_reduction_fraction':1-new/old, 'normalized_bias_budget':target,
        'generic_bound_at_generic_K':increment*float(values[old]),
        'combined_bound_at_combined_K':increment*float(combined[new]),
        'generic_bound_at_combined_K':increment*float(values[new]),
        'structural_bound_at_combined_K':increment*float(structural[new])}, (values,structural)


def select_memories(wall_capacity, theta, increment):
    """Fast sufficient-length selection; exhaustive research audit is separate.

    Bisection returns feasible lengths. It does not require or assert global
    optimality on arbitrary periodic problems. The recorded grid also receives
    an exhaustive integer check, which is not required by the algorithm.
    """
    H=PROTOCOL['horizon'];M,b,P,S,R=arrays(wall_capacity)
    target=PROTOCOL['absolute_contrast_bias_budget_K']/increment
    lipschitz=4/(3*math.sqrt(3))
    def evaluate(L):
        s=H-L;initial=float((P[L]@S[s])[0]);state=float((P[L]@R[s])[0])
        structural=initial+(1/theta+2*S[L,0])*state
        if L:
            h=P[:L]@R[s]
            forcing=b[0]*(np.minimum(h[:,0],2)+(theta+increment)*S[:L,0]*np.minimum(lipschitz*h[:,0],1))
            generic=min(initial+float(P[L-1::-1,0,0]@forcing),2*state/increment,S[H,0]+S[L,0])
        else:generic=S[H,0]
        return generic,structural
    def bisect(column, upper):
        if evaluate(0)[column]<=target:return 0
        low=0;high=upper
        assert evaluate(high)[column]<=target
        while high-low>1:
            mid=(low+high)//2
            if evaluate(mid)[column]<=target:high=mid
            else:low=mid
        return high
    old=bisect(0,H)
    # A structural candidate is useful only when it is no longer than old.
    new=min(old,bisect(1,H))
    assert evaluate(old)[0]<=target and min(evaluate(new))<=target
    return old,new


def assumptions(wall_capacity, zones, theta_max):
    p = PROTOCOL; M = plant(wall_capacity)
    rho = float(max(np.linalg.eigvals(M)))
    v = np.array([1., (rho-M[0,0])/M[0,1]])
    np.testing.assert_allclose(M@v,rho*v,atol=2e-15)
    K = float(v.max()/v.min())
    lowest = 1-p['dt_seconds']/p['air_capacity_J_per_K']*(
        p['air_wall_W_per_K']+2*p['interzone_W_per_K']+p['reference_power_W']*theta_max)
    assert lowest>0 and 0<rho<1 and (v>0).all()
    I=np.eye(zones);ring=np.roll(I,1,axis=1)+np.roll(I,-1,axis=1)-2*I
    full=np.block([[M[0,0]*I+p['dt_seconds']*p['interzone_W_per_K']/p['air_capacity_J_per_K']*ring,M[0,1]*I],
                   [M[1,0]*I,M[1,1]*I]])
    weight=np.concatenate((np.full(zones,v[0]),np.full(zones,v[1])))
    np.testing.assert_allclose(full@weight,rho*weight,atol=2e-15)
    assert full.min()>=0
    np.testing.assert_allclose(np.linalg.norm(full,np.inf),1.)
    return {'rho':rho,'K':K,'positive_weight_air_wall':v.tolist(),
        'minimum_closed_loop_diagonal':lowest,'unweighted_one_step_infinity_norm':1.,
        'state_dimension':2*zones,'saturation_curvature_constant':2.}


def compiled_kernel():
    source = Path(__file__).with_name('thermal_kernel.c')
    extension = '.dylib' if sys.platform=='darwin' else '.so'
    target = ROOT/'build'/('thermal_kernel'+extension); target.parent.mkdir(exist_ok=True)
    start = time.perf_counter()
    command = ['cc','-O3','-std=c99', '-dynamiclib' if sys.platform=='darwin' else '-shared']
    if sys.platform!='darwin': command.append('-fPIC')
    subprocess.run(command+[str(source),'-o',str(target),'-lm'],check=True,capture_output=True)
    elapsed = time.perf_counter()-start
    fn = ctypes.CDLL(str(target)).thermal_contrasts
    pointer = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    fn.argtypes = [pointer,pointer,pointer]+[ctypes.c_int]*4+[ctypes.c_double]*9+[pointer]
    fn.restype = ctypes.c_int
    return fn,elapsed


def simulate(fn, z, wall_capacity, zones, theta, increment, memory, weather, load):
    p = PROTOCOL; out = np.empty(z.shape[0])
    rc = fn(z,weather,load,z.shape[0],z.shape[1],zones,memory,
        p['dt_seconds'],p['air_capacity_J_per_K'],wall_capacity,
        p['air_wall_W_per_K'],p['wall_outdoor_W_per_K'],p['interzone_W_per_K'],
        p['reference_power_W'],theta,increment,out)
    assert rc==0 and np.isfinite(out).all()
    return out


def timed_batch(fn, wall, zones, theta, increment, memory, seed):
    rng = np.random.default_rng(seed); weather,load = schedules()
    weather=np.ascontiguousarray(weather[-memory:]);load=np.ascontiguousarray(load[-memory:])
    start=time.perf_counter(); checksum=0.
    for first in range(0,PROTOCOL['timing_paths'],PROTOCOL['batch_paths']):
        n=min(PROTOCOL['batch_paths'],PROTOCOL['timing_paths']-first)
        z=rng.standard_normal((n,memory,zones+2))
        checksum+=float(simulate(fn,z,wall,zones,theta,increment,memory,weather,load).sum())
    return time.perf_counter()-start,checksum


def verify_kernel(fn):
    """An independent NumPy energy-balance update, including a suffix restart."""
    rng=np.random.default_rng(27182);z=rng.normal(size=(3,13,8))
    weather=np.linspace(3,8,13);load=np.linspace(200,500,13);ca=2e6;cw=8e6
    for L in [1,7,13]:
        got=simulate(fn,z,cw,6,1.2,.01,L,weather,load);ref=[]
        for r in z:
            air=np.zeros((2,6));wall=air.copy()
            for t in range(13-L,13):
                heat=load[t]+150*(.5*r[t,1]+np.sqrt(.75)*r[t,2:])
                outside=weather[t]+1.5*r[t,0]
                aa=air+300/ca*(150*(wall-air)+50*(np.roll(air,1,axis=1)+np.roll(air,-1,axis=1)-2*air)
                    +heat-1000*np.array([1.2,1.21])[:,None]*np.tanh(air))
                wall=wall+300/cw*(150*(air-wall)+100*(outside-wall));air=aa
            ref.append(float(np.mean(air[1]-air[0])))
        np.testing.assert_allclose(got,ref,atol=2e-15,rtol=2e-11)
    return True


def replay_check():
    fn,_=compiled_kernel();assert verify_kernel(fn)
    report=json.loads((EVIDENCE/'thermal_application.json').read_text())
    record=report['records'][0];zones=record['zones']
    rng=np.random.default_rng(PROTOCOL['accuracy_seed'])
    z=rng.standard_normal((2,PROTOCOL['horizon'],zones+2));weather,load=schedules()
    memories=[PROTOCOL['horizon'],record['generic_memory'],record['combined_memory']]
    actual=np.column_stack([simulate(fn,z,record['wall_capacity'],zones,record['theta'],record['increment'],L,weather,load) for L in memories])
    expected=np.load(DATA/'thermal_observations.npz')[record['case']][:2]
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-10)
    return {'passed':True,'independent_numpy_kernel_check':True,'full_horizon_paths_replayed':2,
            'maximum_absolute_replay_difference':float(np.max(abs(actual-expected)))}


def run():
    DATA.mkdir(exist_ok=True);EVIDENCE.mkdir(exist_ok=True)
    # Persist the complete condition grid and comparator definitions before observations.
    (DATA/'thermal_protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    fn,compile_seconds=compiled_kernel();assert verify_kernel(fn)
    weather,load=schedules();raw={};records=[];bound_cache={}
    for wi,wall in enumerate(PROTOCOL['wall_capacities_J_per_K']):
        for ti,theta in enumerate(PROTOCOL['gains']):
            for di,increment in enumerate(PROTOCOL['increments']):
                preparation_trials=[]
                for repeat in range(5):
                    start=time.perf_counter();selected=select_memories(wall,theta,increment)
                    preparation_trials.append(time.perf_counter()-start)
                prep=float(np.median(preparation_trials))
                start=time.perf_counter();bounds,curves=bound_vectors(wall,theta,increment)
                audit_seconds=time.perf_counter()-start
                assert selected==(bounds['generic_memory'],bounds['combined_memory'])
                bounds['exhaustive_bound_audit_seconds']=audit_seconds
                bound_cache[(wi,ti,di)]=(bounds,prep)
                raw[f'bounds_{wi}_{ti}_{di}']=np.column_stack(curves)
                print(json.dumps({'bound_grid':[wi,ti,di],**bounds}),flush=True)
    for ni,zones in enumerate(PROTOCOL['zones']):
        for wi,wall in enumerate(PROTOCOL['wall_capacities_J_per_K']):
            for ti,theta in enumerate(PROTOCOL['gains']):
                # Matched full-horizon tapes for all contrasts and both increments.
                estimates={di:[] for di in range(len(PROTOCOL['increments']))}
                rng=np.random.default_rng(PROTOCOL['accuracy_seed']+100*ni+10*wi+ti)
                for first in range(0,PROTOCOL['accuracy_paths'],PROTOCOL['batch_paths']):
                    count=min(PROTOCOL['batch_paths'],PROTOCOL['accuracy_paths']-first)
                    z=rng.standard_normal((count,PROTOCOL['horizon'],zones+2))
                    for di,d in enumerate(PROTOCOL['increments']):
                        bounds,_=bound_cache[(wi,ti,di)]
                        methods=[PROTOCOL['horizon'],bounds['generic_memory'],bounds['combined_memory']]
                        outputs={L:simulate(fn,z,wall,zones,theta,d,L,weather,load) for L in set(methods)}
                        estimates[di].append(np.column_stack([outputs[L] for L in methods]))
                for di,d in enumerate(PROTOCOL['increments']):
                    key=f'case_{ni}_{wi}_{ti}_{di}';values=np.concatenate(estimates[di]);raw[key]=values
                    bounds,prep=bound_cache[(wi,ti,di)]
                    old,new=bounds['generic_memory'],bounds['combined_memory']
                    timings=[]
                    # Paired, alternating order; each method generates only its retained inputs.
                    for repeat in range(PROTOCOL['timing_repetitions']):
                        order=[0,1] if repeat%2==0 else [1,0];row=[0.,0.]
                        for method in order:
                            row[method],_=timed_batch(fn,wall,zones,theta,d,[old,new][method],
                                PROTOCOL['timing_seed']+10000*ni+1000*wi+100*ti+10*di+repeat)
                        timings.append(row)
                    times=np.array(timings);raw[key+'_times']=times
                    record={'case':key,'zones':zones,'wall_capacity':wall,'theta':theta,'increment':d,
                        **assumptions(wall,zones,theta+d),**bounds,
                        'full_contrast_mean_K':float(values[:,0].mean()),
                        'full_contrast_se_K':float(values[:,0].std(ddof=1)/np.sqrt(len(values))),
                        'generic_mean_truncation_error_K':float((values[:,1]-values[:,0]).mean()),
                        'combined_mean_truncation_error_K':float((values[:,2]-values[:,0]).mean()),
                        'combined_rms_truncation_error_K':float(np.sqrt(np.mean((values[:,2]-values[:,0])**2))),
                        'generic_median_sampling_seconds':float(np.median(times[:,0])),
                        'combined_median_sampling_seconds':float(np.median(times[:,1])),
                        'median_sampling_time_ratio':float(np.median(times[:,1])/np.median(times[:,0])),
                        'bound_preparation_seconds':prep,'shared_compile_seconds':compile_seconds,
                        'preparation_time_repetitions':preparation_trials,
                        'median_total_cost_ratio':float((compile_seconds+prep+np.median(times[:,1]))/
                            (compile_seconds+prep+np.median(times[:,0]))),
                        'median_cost_ratio_with_compiled_kernel':float((prep+np.median(times[:,1]))/
                            (prep+np.median(times[:,0])))}
                    records.append(record);print(json.dumps(record),flush=True)
    np.savez_compressed(DATA/'thermal_observations.npz',**raw)
    report={'protocol':PROTOCOL,'records':records,'independent_kernel_check':True,
        'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
        'cost_scope':'RNG and all simulated states included; same compile and complete comparator preparation charged to each method. Imports and disk serialization excluded.',
        'timing_note':'New measurements; alternating order, five paired repetitions. No claim of platform-independent speedup.',
        'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ['thermal_application.py','thermal_kernel.c']}}
    (EVIDENCE/'thermal_application.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(verify(),indent=2))


def remeasure_preparation():
    """Retain all observations, measure the implemented fast selector separately."""
    path=EVIDENCE/'thermal_application.json';report=json.loads(path.read_text())
    cache={}
    for r in report['records']:
        key=(r['wall_capacity'],r['theta'],r['increment'])
        if key not in cache:
            trials=[]
            for repeat in range(5):
                start=time.perf_counter();got=select_memories(*key);trials.append(time.perf_counter()-start)
            assert got==(r['generic_memory'],r['combined_memory'])
            cache[key]=trials
        r['exhaustive_bound_audit_seconds']=r['bound_preparation_seconds']
        r['preparation_time_repetitions']=cache[key]
        r['bound_preparation_seconds']=float(np.median(cache[key]))
        before=r['generic_median_sampling_seconds'];after=r['combined_median_sampling_seconds']
        prep=r['bound_preparation_seconds'];compile_seconds=r['shared_compile_seconds']
        r['median_total_cost_ratio']=(compile_seconds+prep+after)/(compile_seconds+prep+before)
        r['median_cost_ratio_with_compiled_kernel']=(prep+after)/(prep+before)
    report['preparation_timing_note']='Fast feasible-length search timed separately in five repetitions after observations; exhaustive integer audit excluded as research validation. All original simulation timings retained.'
    report['source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ['thermal_application.py','thermal_kernel.c']}
    path.write_text(json.dumps(report,indent=2));print(json.dumps(verify(),indent=2))


def verify():
    report=json.loads((EVIDENCE/'thermal_application.json').read_text())
    assert report['protocol']==PROTOCOL
    raw=np.load(DATA/'thermal_observations.npz');cache={};small=[];large=[]
    for r in report['records']:
        key=(r['wall_capacity'],r['theta'],r['increment'])
        if key not in cache:cache[key]=bound_vectors(*key)
        b,curves=cache[key]
        for k,v in b.items():np.testing.assert_allclose(r[k],v,rtol=2e-13,atol=2e-15)
        wi=PROTOCOL['wall_capacities_J_per_K'].index(key[0]);ti=PROTOCOL['gains'].index(key[1]);di=PROTOCOL['increments'].index(key[2])
        np.testing.assert_allclose(raw[f'bounds_{wi}_{ti}_{di}'],np.column_stack(curves),rtol=2e-13,atol=2e-15)
        assumptions(r['wall_capacity'],r['zones'],r['theta']+r['increment'])
        x=raw[r['case']];t=raw[r['case']+'_times']
        assert x.shape==(PROTOCOL['accuracy_paths'],3) and t.shape==(5,2)
        np.testing.assert_allclose(x[:,0].mean(),r['full_contrast_mean_K'],rtol=2e-13)
        np.testing.assert_allclose(x[:,0].std(ddof=1)/np.sqrt(len(x)),r['full_contrast_se_K'])
        np.testing.assert_allclose((x[:,1]-x[:,0]).mean(),r['generic_mean_truncation_error_K'])
        np.testing.assert_allclose((x[:,2]-x[:,0]).mean(),r['combined_mean_truncation_error_K'])
        np.testing.assert_allclose(np.median(t[:,0]),r['generic_median_sampling_seconds'])
        np.testing.assert_allclose(np.median(t[:,1]),r['combined_median_sampling_seconds'])
        np.testing.assert_allclose(np.median(t[:,1])/np.median(t[:,0]),r['median_sampling_time_ratio'])
        np.testing.assert_allclose(np.sqrt(np.mean((x[:,2]-x[:,0])**2)),r['combined_rms_truncation_error_K'])
        old,new=select_memories(*key)
        assert (old,new)==(r['generic_memory'],r['combined_memory'])
        pre=r['bound_preparation_seconds'];cold=r['shared_compile_seconds']
        np.testing.assert_allclose(np.median(r['preparation_time_repetitions']),pre)
        np.testing.assert_allclose((cold+pre+np.median(t[:,1]))/(cold+pre+np.median(t[:,0])),r['median_total_cost_ratio'])
        np.testing.assert_allclose((pre+np.median(t[:,1]))/(pre+np.median(t[:,0])),r['median_cost_ratio_with_compiled_kernel'])
        assert r['combined_bound_at_combined_K']<=PROTOCOL['absolute_contrast_bias_budget_K']
        (small if r['increment']==.01 else large).append(r)
    assert len(small)==len(large)==12 and len(raw.files)==60
    return {'passed':True,'cases':24,'accuracy_paths_per_case':PROTOCOL['accuracy_paths'],
        'small_increment_step_reduction_range':[min(r['step_reduction_fraction'] for r in small),max(r['step_reduction_fraction'] for r in small)],
        'small_increment_sampling_time_ratio_range':[min(r['median_sampling_time_ratio'] for r in small),max(r['median_sampling_time_ratio'] for r in small)],
        'small_increment_total_cost_ratio_range':[min(r['median_total_cost_ratio'] for r in small),max(r['median_total_cost_ratio'] for r in small)],
        'small_increment_compiled_cost_ratio_range':[min(r['median_cost_ratio_with_compiled_kernel'] for r in small),max(r['median_cost_ratio_with_compiled_kernel'] for r in small)],
        'large_increment_step_reduction_range':[min(r['step_reduction_fraction'] for r in large),max(r['step_reduction_fraction'] for r in large)],
        'largest_observed_rms_error_K':max(r['combined_rms_truncation_error_K'] for r in report['records']),
        'scope':'Empirical error summaries accompany a proven model bound; timing ratios are descriptive.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--verify',action='store_true');parser.add_argument('--remeasure-preparation',action='store_true');parser.add_argument('--check-kernel',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    elif args.remeasure_preparation:remeasure_preparation()
    elif args.check_kernel:print(json.dumps(replay_check(),indent=2))
    elif args.verify:print(json.dumps(verify(),indent=2))
    else:parser.print_help()
