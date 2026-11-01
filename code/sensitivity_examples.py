"""Deterministic sensitivity-forgetting examples with independent complex-step checks."""
from pathlib import Path
import csv, json, math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def nonlinearities():
    return [
        ('tanh', np.tanh, lambda x: 1-np.tanh(x)**2, 2., 1., 4/(3*math.sqrt(3))),
        ('arctan', lambda x: 2/math.pi*np.arctan(x),
         lambda x: 2/math.pi/(1+x*x), 1., 2/math.pi, 9/(4*math.pi*math.sqrt(3))),
        ('smooth_softsign', lambda x: x/np.sqrt(1+x*x),
         lambda x: (1+x*x)**(-1.5), 1.5, 1., 1.5/(1.25**2.5))]

def main():
    rows, summaries = [], []
    theta = np.array([0.07, 0.11]); direction = np.array([1., -0.4])
    initial = np.array([3., -2.]); maximum_complex_error = 0.
    increment = 0.02; theta_new = theta+increment*direction
    for name, A, C, K, rho in [
        ('mixing', np.array([[0.55, 0.15], [0.10, 0.60]]),
         np.array([[0.8, 0.2], [0.1, 0.9]]), 1., 0.7),
        ('transient', np.array([[0.5, 2.], [0., 0.5]]), np.eye(2), 4., 0.75)]:
        for label, g, gp, curvature, Lg, Lgp in nonlinearities():
            x = initial.copy(); y = np.zeros(2); q = np.zeros(2); p = np.zeros(2)
            h = abs(initial); envelope = np.eye(2); generic_sum = np.zeros(2)
            xc = initial.astype(complex); yc = np.zeros(2, dtype=complex)
            x_new = initial.copy(); y_new = np.zeros(2)
            complex_theta = theta+1e-20j*direction
            observed, generic, structural, complex_error, matrix_norms = [], [], [], [], [1.]
            finite_fractions = []
            for t in range(32):
                cx, cy = C@x, C@y; gx, gy = g(cx), g(cy)
                sec = np.divide(gx-gy, cx-cy, out=gp(cx).copy(), where=abs(cx-cy)>1e-10)
                M = A-theta[:, None]*sec[:, None]*C
                Jx = A-theta[:, None]*gp(cx)[:, None]*C
                Jy = A-theta[:, None]*gp(cy)[:, None]*C
                assert M.min() >= -1e-14 and Jx.min() >= -1e-14 and Jy.min() >= -1e-14
                u = 0.12*np.array([math.sin(0.7*(t+1)), math.cos(0.5*(t+1))])
                generic_sum = A@generic_sum+C@(envelope@abs(initial))
                h = M@h; envelope = A@envelope
                q, p = Jx@q-direction*gx, Jy@p-direction*gy
                x, y = A@x-theta*gx+u, A@y-theta*gy+u
                x_new, y_new = A@x_new-theta_new*g(C@x_new)+u, A@y_new-theta_new*g(C@y_new)+u
                xc, yc = A@xc-complex_theta*g(C@xc)+u, A@yc-complex_theta*g(C@yc)+u
                L = t+1; S = K*(1-rho**L)/(1-rho)
                kappa = 1/theta.min()+curvature*S
                defect = envelope@abs(initial)-h
                assert defect.min() > -1e-12
                structural_vector = kappa*defect
                generic_vector = (Lg+theta.max()*Lgp*S)*generic_sum
                actual_vector = abs(q-p)
                assert np.all(actual_vector <= structural_vector+2e-12)
                assert np.all(actual_vector <= generic_vector+2e-12)
                np.testing.assert_allclose((xc-yc).imag/1e-20, q-p, rtol=2e-8, atol=2e-12)
                error = float(np.max(abs((xc-yc).imag/1e-20-(q-p))))
                maximum_complex_error = max(maximum_complex_error, error)
                actual = float(np.max(actual_vector)); new = float(np.max(structural_vector))
                old = float(np.max(generic_vector)); norm = float(np.linalg.norm(envelope, np.inf))
                assert norm <= K*rho**L*(1+1e-12)
                observed.append(actual); structural.append(new); generic.append(old)
                complex_error.append(error); matrix_norms.append(norm)
                finite_error = abs(((x_new-x)-(y_new-y))[0])/increment
                finite_bound = kappa*(envelope@abs(initial))[0]
                assert finite_error <= finite_bound+2e-11
                finite_fractions.append(float(finite_error/finite_bound))
                rows.append({'system':name,'nonlinearity':label,'L':L,'K':K,'rho':rho,
                    'actual_tangent_difference':actual,'common_defect_bound':new,
                    'separate_lipschitz_bound':old,'plant_product_norm':norm,
                    'complex_step_absolute_error':error,'finite_gain_output_error':float(finite_error),
                    'finite_gain_output_bound':float(finite_bound)})
            summaries.append({'system':name,'nonlinearity':label,'K':K,'rho':rho,
                'maximum_actual_over_common_bound':max(a/b for a,b in zip(observed,structural)),
                'common_over_generic_at_32':structural[-1]/generic[-1],
                'maximum_finite_gain_output_fraction':max(finite_fractions),
                'maximum_plant_product_norm':max(matrix_norms)})
    gain = []
    for relative in np.geomspace(1e-5, 0.2, 45):
        loga = math.log1p(-float(relative)); optimum = -1/loga
        candidates = {max(1,math.floor(optimum)),max(1,math.ceil(optimum))}
        scaled = max(relative*L*math.exp((L-1)*loga) for L in candidates)
        gain.append({'relative_gain':float(relative),'theta_times_supremum':scaled,
                     'candidate_lengths':sorted(candidates)})
    assert abs(gain[0]['theta_times_supremum']-math.exp(-1)) < 3e-6
    result = {'passed':True,'deterministic_cases':len(summaries),'lengths_per_case':32,
        'theta':theta.tolist(),'gain_direction':direction.tolist(),'initial_state':initial.tolist(),
        'finite_gain_increment':increment,
        'maximum_complex_step_absolute_error':maximum_complex_error,
        'summaries':summaries,'scalar_gain_scaling':gain,
        'scope':'Deterministic illustrations and numerical identity checks. They are not new stochastic performance timings, calibrated physical tests, or directed-rounding certificates.'}
    (ROOT/'evidence').mkdir(exist_ok=True);(ROOT/'recorded_data').mkdir(exist_ok=True)
    (ROOT/'evidence/sensitivity_examples.json').write_text(json.dumps(result,indent=2))
    with (ROOT/'recorded_data/sensitivity_examples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ['scalar_gain_scaling']},indent=2))

if __name__=='__main__':
    main()
