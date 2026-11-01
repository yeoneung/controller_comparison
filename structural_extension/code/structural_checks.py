"""Checks for unbounded sector nonlinearities, log gains, and critical modes."""
from pathlib import Path
import csv,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def run():
    A=np.array([[.55,.1],[.05,.6]]);C=np.array([[.8,.2],[.1,.9]])
    weights=np.array([1.,4.]);W=np.diag(weights);Winv=np.diag(1/weights)
    original_A=W@A@Winv;original_B=W;original_C=C@Winv
    rho=.65;input_bound=.12;gain_upper=.2
    functions=[('linear',lambda z:z,lambda z:np.ones_like(z),1.,0.),
        ('linear_plus_tanh',lambda z:z+.25*np.tanh(z),lambda z:1+.25*(1-np.tanh(z)**2),1.25,.2),
        ('centered_softplus',lambda z:np.log1p(np.exp(z))-np.log(2),lambda z:1/(1+np.exp(-z)),1.,1.)]
    rows=[];summaries=[];maximum_complex_error=0.
    for name,g,gp,slope,curvature in functions:
        S=gain_upper*slope*input_bound/(1-rho)**2;kappa=1+curvature*S
        for gain in [1e-4,.02,.1]:
            a=np.full(2,gain);direction=np.array([1.,-.4]);d=.05
            x=np.array([3.,-2.]);y=np.zeros(2);q=y.copy();p=y.copy()
            xc=x.astype(complex);yc=y.astype(complex);ac=a*np.exp(1e-20j*direction)
            xn=x.copy();yn=y.copy();an=a*np.exp(d*direction)
            h0=abs(x);h=h0.copy();P=np.eye(2);ratios=[];finite=[]
            x_original=W@x
            for t in range(80):
                u=.12*np.array([np.sin(.7*(t+1)),np.cos(.5*(t+1))])
                cx=C@x;cy=C@y;gx=g(cx);gy=g(cy)
                sec=np.divide(gx-gy,cx-cy,out=gp(cx).copy(),where=abs(cx-cy)>1e-10)
                M=A-a[:,None]*sec[:,None]*C
                Jx=A-a[:,None]*gp(cx)[:,None]*C;Jy=A-a[:,None]*gp(cy)[:,None]*C
                assert M.min()>=0 and Jx.min()>=0 and Jy.min()>=0
                q,p=Jx@q-a*direction*gx,Jy@p-a*direction*gy
                h=M@h;P=A@P
                x,y=A@x-a*gx+u,A@y-a*gy+u
                xn,yn=A@xn-an*g(C@xn)+u,A@yn-an*g(C@yn)+u
                xc,yc=A@xc-ac*g(C@xc)+u,A@yc-ac*g(C@yc)+u
                x_original=original_A@x_original-original_B@(a*g(original_C@x_original))+W@u
                np.testing.assert_allclose(x_original,W@x,atol=3e-15)
                actual=abs(q-p);bound=kappa*(P@h0-h)
                assert np.all(actual<=bound+1e-12)
                error=float(np.max(abs((xc-yc).imag/1e-20-(q-p))))
                maximum_complex_error=max(maximum_complex_error,error)
                np.testing.assert_allclose((xc-yc).imag/1e-20,q-p,atol=1e-12,rtol=1e-7)
                finite_error=float(abs(((xn-x)-(yn-y))[0])/d)
                finite_bound=float(kappa*(P@h0)[0])
                assert finite_error<=finite_bound+1e-12
                ratios.append(float(np.max(actual)/max(np.max(bound),1e-280)))
                finite.append(finite_error/max(finite_bound,1e-280))
                rows.append({'nonlinearity':name,'gain':gain,'length':t+1,'chi':1.,'kappa':kappa,
                    'weighted_tangent_error':float(max(actual)),'weighted_defect_bound':float(max(bound)),
                    'finite_log_gain_error':finite_error,'finite_log_gain_bound':finite_bound})
            summaries.append({'nonlinearity':name,'gain':gain,'kappa_uniform_in_gain':kappa,
                'maximum_error_fraction':max(ratios),'maximum_finite_gain_fraction':max(finite)})
    # A primitive three-dimensional plant; a non-scalar active feedback direction.
    rho=.8;A=rho*np.array([[.75,.2,.05],[.1,.75,.15],[.2,.1,.7]])
    E=np.diag([1.,.3,.5]);eigen,left=np.linalg.eig(A.T)
    w=left[:,np.argmin(abs(eigen-rho))].real;w/=w.sum();critical_coupling=float(w@E@np.ones(3))
    matrix=[]
    for a in [.02,.01,.003,.001,.0003,.0001]:
        N=(A-a*E)/rho;power=np.eye(3);Z=np.zeros((3,3));maximum=0.;argmax=0
        cap=math.ceil(8*rho/(a*critical_coupling))
        for L in range(1,cap+1):
            Z=N@Z+(E/rho)@power;power=N@power
            value=float(np.linalg.norm(Z,np.inf))
            if value>maximum:maximum=value;argmax=L
        checkL=37;derivative=np.linalg.matrix_power(N+1e-20j*E/rho,checkL).imag/1e-20
        z=np.zeros((3,3));p=np.eye(3)
        for _ in range(checkL):z=N@z+(E/rho)@p;p=N@p
        np.testing.assert_allclose(derivative,z,rtol=2e-12,atol=2e-12)
        matrix.append({'gain':a,'searched_lengths':cap,'maximizing_length':argmax,
            'absolute_gain_constant':maximum,'log_gain_constant':a*maximum,
            'square_gain_parameter_constant':math.sqrt(a)*maximum})
    assert abs(matrix[-1]['log_gain_constant']-math.exp(-1))<2e-4
    report={'passed':True,'sector_nonlinearities':3,'gain_values':3,'lengths_per_case':80,
        'bounded_nonlinearity_assumed':False,'bounded_inputs':True,
        'weighted_norm_weights':weights.tolist(),'maximum_complex_step_error':maximum_complex_error,
        'sector_summaries':summaries,'matrix_gain_characterization':matrix,
        'critical_coupling':critical_coupling,
        'inactive_mode_example':'M=diag(0.8,0.5), E=diag(0,1): no critical self-coupling; absolute-gain sensitivity remains bounded at rate 0.8.',
        'scope':'Numerical illustrations of analytic arguments, not historical novelty or validated rounding.'}
    (ROOT/'evidence').mkdir(exist_ok=True);(ROOT/'recorded_data').mkdir(exist_ok=True)
    (ROOT/'evidence/structural_checks.json').write_text(json.dumps(report,indent=2))
    with (ROOT/'recorded_data/structural_checks.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(report,indent=2))

if __name__=='__main__':run()
