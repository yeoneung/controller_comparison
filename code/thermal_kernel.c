#include <math.h>
#include <stdlib.h>

/* Two-capacitance zones on a ring; identical kernel for every memory. */
int thermal_contrasts(const double *z, const double *weather,
    const double *load, int paths, int horizon, int zones, int memory,
    double dt, double air_capacity, double wall_capacity,
    double air_wall, double wall_outdoor, double interzone,
    double reference_power, double theta, double increment,
    double *out) {
    double *storage = calloc((size_t)8 * zones, sizeof(double));
    if (!storage) return 1;
    const double ba = dt / air_capacity, bw = dt / wall_capacity;
    for (int p = 0; p < paths; ++p) {
        for (int j = 0; j < 8 * zones; ++j) storage[j] = 0.;
        double *a = storage, *w = a + zones;
        double *an = w + zones, *wn = an + zones;
        double *b = wn + zones, *v = b + zones;
        double *bn = v + zones, *vn = bn + zones;
        for (int t = horizon - memory; t < horizon; ++t) {
            const double *r = z + ((size_t)p * horizon + t) * (zones + 2);
            const double outside = weather[t] + 1.5 * r[0];
            for (int j = 0; j < zones; ++j) {
                const int left = (j + zones - 1) % zones;
                const int right = (j + 1) % zones;
                const double heat = load[t] + 150. * (0.5 * r[1]
                    + sqrt(0.75) * r[j + 2]);
                an[j] = a[j] + ba * (air_wall * (w[j] - a[j])
                    + interzone * (a[left] + a[right] - 2. * a[j])
                    + heat - reference_power * theta * tanh(a[j]));
                wn[j] = w[j] + bw * (air_wall * (a[j] - w[j])
                    + wall_outdoor * (outside - w[j]));
                bn[j] = b[j] + ba * (air_wall * (v[j] - b[j])
                    + interzone * (b[left] + b[right] - 2. * b[j])
                    + heat - reference_power * (theta + increment) * tanh(b[j]));
                vn[j] = v[j] + bw * (air_wall * (b[j] - v[j])
                    + wall_outdoor * (outside - v[j]));
            }
            for (int j = 0; j < zones; ++j) {
                a[j] = an[j]; w[j] = wn[j]; b[j] = bn[j]; v[j] = vn[j];
            }
        }
        double difference = 0.;
        for (int j = 0; j < zones; ++j) difference += b[j] - a[j];
        out[p] = difference / zones;
    }
    free(storage);
    return 0;
}
