#include <math.h>
#include <stdlib.h>

int gain_sweep(const double *z, const double *weather, const double *load,
    int paths, int steps, int zones, int comparisons, double dt, double ca,
    double cw, double gaw, double go, double gz, double qref, double increment,
    const double *gains, double *out) {
    size_t count = (size_t)comparisons * zones;
    double *storage = calloc(8 * count + zones, sizeof(double));
    if (!storage) return 1;
    double *heat = storage + 8 * count;
    for (int p = 0; p < paths; ++p) {
        for (size_t j = 0; j < 8 * count; ++j) storage[j] = 0.;
        double *a=storage, *w=a+count, *an=w+count, *wn=an+count;
        double *b=wn+count, *v=b+count, *bn=v+count, *vn=bn+count;
        for (int t = 0; t < steps; ++t) {
            const double *r=z+((size_t)p*steps+t)*(zones+2);
            double outside=weather[t]+1.5*r[0];
            for (int j=0; j<zones; ++j)
                heat[j]=load[t]+150.*(.5*r[1]+sqrt(.75)*r[j+2]);
            for (int k=0; k<comparisons; ++k) {
                size_t offset=(size_t)k*zones;
                for (int j=0; j<zones; ++j) {
                    size_t i=offset+j, left=offset+(j+zones-1)%zones;
                    size_t right=offset+(j+1)%zones;
                    an[i]=a[i]+dt/ca*(gaw*(w[i]-a[i])
                        +gz*(a[left]+a[right]-2*a[i])+heat[j]
                        -qref*gains[k]*tanh(a[i]));
                    wn[i]=w[i]+dt/cw*(gaw*(a[i]-w[i])+go*(outside-w[i]));
                    bn[i]=b[i]+dt/ca*(gaw*(v[i]-b[i])
                        +gz*(b[left]+b[right]-2*b[i])+heat[j]
                        -qref*(gains[k]+increment)*tanh(b[i]));
                    vn[i]=v[i]+dt/cw*(gaw*(b[i]-v[i])+go*(outside-v[i]));
                }
            }
            double *tmp=a;a=an;an=tmp;tmp=w;w=wn;wn=tmp;
            tmp=b;b=bn;bn=tmp;tmp=v;v=vn;vn=tmp;
        }
        for (int k=0; k<comparisons; ++k) {
            double total=0.;
            for (int j=0;j<zones;++j) total+=b[(size_t)k*zones+j]-a[(size_t)k*zones+j];
            out[(size_t)p*comparisons+k]=total/zones;
        }
    }
    free(storage);
    return 0;
}
