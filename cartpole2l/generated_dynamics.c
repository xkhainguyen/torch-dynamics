/* This file was automatically generated by CasADi 3.6.3.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) generated_dynamics_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};

/* eval_forward_dynamics:(i0[3],i1[3],i2[3],i3)->(o0[3],o1[3]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][0] : 0;
  a1=arg[3]? arg[3][0] : 0;
  a2=arg[1]? arg[1][0] : 0;
  a3=2.;
  a4=(a1/a3);
  a5=arg[2]? arg[2][0] : 0;
  a6=arg[0]? arg[0][1] : 0;
  a7=cos(a6);
  a8=arg[1]? arg[1][1] : 0;
  a9=sin(a6);
  a10=(a9*a2);
  a11=(a8*a10);
  a12=arg[0]? arg[0][2] : 0;
  a13=cos(a12);
  a14=arg[1]? arg[1][2] : 0;
  a15=(a14+a8);
  a16=sin(a12);
  a17=(a7*a2);
  a18=(a17-a8);
  a19=(a16*a18);
  a20=(a13*a10);
  a19=(a19+a20);
  a20=(a15*a19);
  a21=-5.0000000000000000e-01;
  a22=arg[2]? arg[2][2] : 0;
  a23=(a19*a15);
  a23=(a22+a23);
  a24=(a21*a23);
  a25=5.0000000000000000e-01;
  a19=(a14*a19);
  a26=(a25*a19);
  a24=(a24-a26);
  a20=(a20+a24);
  a24=(a13*a20);
  a18=(a13*a18);
  a26=(a16*a10);
  a18=(a18-a26);
  a26=(a18-a15);
  a15=(a15*a26);
  a18=(a14*a18);
  a15=(a15-a18);
  a18=(a16*a15);
  a24=(a24-a18);
  a11=(a11+a24);
  a18=-1.;
  a26=(a25*a13);
  a27=(a26*a13);
  a28=casadi_sq(a16);
  a27=(a27+a28);
  a28=(a18-a27);
  a29=(a3+a27);
  a30=(a28/a29);
  a31=arg[2]? arg[2][1] : 0;
  a24=(a22-a24);
  a32=(a10*a8);
  a24=(a24-a32);
  a24=(a31-a24);
  a32=(a30*a24);
  a33=1.;
  a27=(a33+a27);
  a34=(a30*a28);
  a27=(a27-a34);
  a10=(a8*a10);
  a34=(a27*a10);
  a26=(a26*a16);
  a35=(a16*a13);
  a26=(a26-a35);
  a35=(a30*a26);
  a35=(a26+a35);
  a36=(a8*a17);
  a37=(a35*a36);
  a34=(a34+a37);
  a32=(a32-a34);
  a11=(a11+a32);
  a11=(a7*a11);
  a17=(a17-a8);
  a17=(a8*a17);
  a20=(a16*a20);
  a15=(a13*a15);
  a20=(a20+a15);
  a17=(a17+a20);
  a20=(a25*a16);
  a15=(a20*a13);
  a32=(a13*a16);
  a15=(a15-a32);
  a32=(a26/a29);
  a28=(a32*a28);
  a15=(a15+a28);
  a28=(a15*a10);
  a20=(a20*a16);
  a34=casadi_sq(a13);
  a20=(a20+a34);
  a20=(a33+a20);
  a26=(a32*a26);
  a20=(a20-a26);
  a26=(a20*a36);
  a28=(a28+a26);
  a26=(a32*a24);
  a28=(a28+a26);
  a17=(a17-a28);
  a17=(a9*a17);
  a11=(a11-a17);
  a11=(a5-a11);
  a17=10.;
  a28=(a7*a27);
  a26=(a9*a15);
  a28=(a28-a26);
  a28=(a28*a7);
  a26=(a7*a35);
  a34=(a9*a20);
  a26=(a26-a34);
  a26=(a26*a9);
  a28=(a28-a26);
  a28=(a17+a28);
  a11=(a11/a28);
  a26=9.8100000000000005e+00;
  a27=(a9*a27);
  a15=(a7*a15);
  a27=(a27+a15);
  a27=(a27*a7);
  a35=(a9*a35);
  a20=(a7*a20);
  a35=(a35+a20);
  a35=(a35*a9);
  a27=(a27-a35);
  a27=(a27/a28);
  a27=(a26*a27);
  a11=(a11-a27);
  a27=(a4*a11);
  a27=(a2+a27);
  a27=(a3*a27);
  a27=(a2+a27);
  a28=(a1/a3);
  a35=(a1/a3);
  a20=(a35*a8);
  a20=(a6+a20);
  a15=cos(a20);
  a34=(a1/a3);
  a24=(a24/a29);
  a29=(a7*a11);
  a37=(a26*a9);
  a29=(a29+a37);
  a29=(a29-a10);
  a30=(a30*a29);
  a7=(a26*a7);
  a9=(a9*a11);
  a7=(a7-a9);
  a7=(a7-a36);
  a32=(a32*a7);
  a30=(a30-a32);
  a24=(a24-a30);
  a30=(a34*a24);
  a30=(a8+a30);
  a20=sin(a20);
  a32=(a34*a11);
  a32=(a2+a32);
  a36=(a20*a32);
  a9=(a30*a36);
  a35=(a35*a14);
  a35=(a12+a35);
  a10=cos(a35);
  a23=(a25*a23);
  a29=(a29-a24);
  a13=(a13*a29);
  a16=(a16*a7);
  a13=(a13+a16);
  a13=(a13-a19);
  a13=(a21*a13);
  a13=(a13+a24);
  a23=(a23-a13);
  a34=(a34*a23);
  a34=(a14+a34);
  a13=(a34+a30);
  a35=sin(a35);
  a32=(a15*a32);
  a19=(a32-a30);
  a16=(a35*a19);
  a7=(a10*a36);
  a16=(a16+a7);
  a7=(a13*a16);
  a29=(a16*a13);
  a29=(a22+a29);
  a37=(a21*a29);
  a16=(a34*a16);
  a38=(a25*a16);
  a37=(a37-a38);
  a7=(a7+a37);
  a37=(a10*a7);
  a19=(a10*a19);
  a38=(a35*a36);
  a19=(a19-a38);
  a38=(a19-a13);
  a13=(a13*a38);
  a34=(a34*a19);
  a13=(a13-a34);
  a34=(a35*a13);
  a37=(a37-a34);
  a9=(a9+a37);
  a34=(a25*a10);
  a19=(a34*a10);
  a38=casadi_sq(a35);
  a19=(a19+a38);
  a38=(a18-a19);
  a39=(a3+a19);
  a40=(a38/a39);
  a37=(a22-a37);
  a41=(a36*a30);
  a37=(a37-a41);
  a37=(a31-a37);
  a41=(a40*a37);
  a19=(a33+a19);
  a42=(a40*a38);
  a19=(a19-a42);
  a36=(a30*a36);
  a42=(a19*a36);
  a34=(a34*a35);
  a43=(a35*a10);
  a34=(a34-a43);
  a43=(a40*a34);
  a43=(a34+a43);
  a44=(a30*a32);
  a45=(a43*a44);
  a42=(a42+a45);
  a41=(a41-a42);
  a9=(a9+a41);
  a9=(a15*a9);
  a32=(a32-a30);
  a30=(a30*a32);
  a7=(a35*a7);
  a13=(a10*a13);
  a7=(a7+a13);
  a30=(a30+a7);
  a7=(a25*a35);
  a13=(a7*a10);
  a32=(a10*a35);
  a13=(a13-a32);
  a32=(a34/a39);
  a38=(a32*a38);
  a13=(a13+a38);
  a38=(a13*a36);
  a7=(a7*a35);
  a41=casadi_sq(a10);
  a7=(a7+a41);
  a7=(a33+a7);
  a34=(a32*a34);
  a7=(a7-a34);
  a34=(a7*a44);
  a38=(a38+a34);
  a34=(a32*a37);
  a38=(a38+a34);
  a30=(a30-a38);
  a30=(a20*a30);
  a9=(a9-a30);
  a9=(a5-a9);
  a30=(a15*a19);
  a38=(a20*a13);
  a30=(a30-a38);
  a30=(a30*a15);
  a38=(a15*a43);
  a34=(a20*a7);
  a38=(a38-a34);
  a38=(a38*a20);
  a30=(a30-a38);
  a30=(a17+a30);
  a9=(a9/a30);
  a19=(a20*a19);
  a13=(a15*a13);
  a19=(a19+a13);
  a19=(a19*a15);
  a43=(a20*a43);
  a7=(a15*a7);
  a43=(a43+a7);
  a43=(a43*a20);
  a19=(a19-a43);
  a19=(a19/a30);
  a19=(a26*a19);
  a9=(a9-a19);
  a19=(a28*a9);
  a19=(a2+a19);
  a19=(a3*a19);
  a27=(a27+a19);
  a19=(a1/a3);
  a30=(a4*a24);
  a30=(a8+a30);
  a30=(a19*a30);
  a30=(a6+a30);
  a43=cos(a30);
  a7=(a1/a3);
  a37=(a37/a39);
  a39=(a15*a9);
  a13=(a26*a20);
  a39=(a39+a13);
  a39=(a39-a36);
  a40=(a40*a39);
  a15=(a26*a15);
  a20=(a20*a9);
  a15=(a15-a20);
  a15=(a15-a44);
  a32=(a32*a15);
  a40=(a40-a32);
  a37=(a37-a40);
  a40=(a7*a37);
  a40=(a8+a40);
  a30=sin(a30);
  a32=(a7*a9);
  a32=(a2+a32);
  a44=(a30*a32);
  a20=(a40*a44);
  a36=(a4*a23);
  a36=(a14+a36);
  a19=(a19*a36);
  a19=(a12+a19);
  a36=cos(a19);
  a29=(a25*a29);
  a39=(a39-a37);
  a10=(a10*a39);
  a35=(a35*a15);
  a10=(a10+a35);
  a10=(a10-a16);
  a10=(a21*a10);
  a10=(a10+a37);
  a29=(a29-a10);
  a7=(a7*a29);
  a7=(a14+a7);
  a10=(a7+a40);
  a19=sin(a19);
  a32=(a43*a32);
  a16=(a32-a40);
  a35=(a19*a16);
  a15=(a36*a44);
  a35=(a35+a15);
  a15=(a10*a35);
  a39=(a35*a10);
  a39=(a22+a39);
  a13=(a21*a39);
  a35=(a7*a35);
  a38=(a25*a35);
  a13=(a13-a38);
  a15=(a15+a13);
  a13=(a36*a15);
  a16=(a36*a16);
  a38=(a19*a44);
  a16=(a16-a38);
  a38=(a16-a10);
  a10=(a10*a38);
  a7=(a7*a16);
  a10=(a10-a7);
  a7=(a19*a10);
  a13=(a13-a7);
  a20=(a20+a13);
  a7=(a25*a36);
  a16=(a7*a36);
  a38=casadi_sq(a19);
  a16=(a16+a38);
  a38=(a18-a16);
  a34=(a3+a16);
  a41=(a38/a34);
  a13=(a22-a13);
  a42=(a44*a40);
  a13=(a13-a42);
  a13=(a31-a13);
  a42=(a41*a13);
  a16=(a33+a16);
  a45=(a41*a38);
  a16=(a16-a45);
  a44=(a40*a44);
  a45=(a16*a44);
  a7=(a7*a19);
  a46=(a19*a36);
  a7=(a7-a46);
  a46=(a41*a7);
  a46=(a7+a46);
  a47=(a40*a32);
  a48=(a46*a47);
  a45=(a45+a48);
  a42=(a42-a45);
  a20=(a20+a42);
  a20=(a43*a20);
  a32=(a32-a40);
  a40=(a40*a32);
  a15=(a19*a15);
  a10=(a36*a10);
  a15=(a15+a10);
  a40=(a40+a15);
  a15=(a25*a19);
  a10=(a15*a36);
  a32=(a36*a19);
  a10=(a10-a32);
  a32=(a7/a34);
  a38=(a32*a38);
  a10=(a10+a38);
  a38=(a10*a44);
  a15=(a15*a19);
  a42=casadi_sq(a36);
  a15=(a15+a42);
  a15=(a33+a15);
  a7=(a32*a7);
  a15=(a15-a7);
  a7=(a15*a47);
  a38=(a38+a7);
  a7=(a32*a13);
  a38=(a38+a7);
  a40=(a40-a38);
  a40=(a30*a40);
  a20=(a20-a40);
  a20=(a5-a20);
  a40=(a43*a16);
  a38=(a30*a10);
  a40=(a40-a38);
  a40=(a40*a43);
  a38=(a43*a46);
  a7=(a30*a15);
  a38=(a38-a7);
  a38=(a38*a30);
  a40=(a40-a38);
  a40=(a17+a40);
  a20=(a20/a40);
  a16=(a30*a16);
  a10=(a43*a10);
  a16=(a16+a10);
  a16=(a16*a43);
  a46=(a30*a46);
  a15=(a43*a15);
  a46=(a46+a15);
  a46=(a46*a30);
  a16=(a16-a46);
  a16=(a16/a40);
  a16=(a26*a16);
  a20=(a20-a16);
  a16=(a1*a20);
  a16=(a2+a16);
  a27=(a27+a16);
  a16=6.;
  a27=(a27/a16);
  a27=(a1*a27);
  a0=(a0+a27);
  if (res[0]!=0) res[0][0]=a0;
  a0=(a4*a24);
  a0=(a8+a0);
  a0=(a3*a0);
  a0=(a8+a0);
  a27=(a28*a37);
  a27=(a8+a27);
  a27=(a3*a27);
  a0=(a0+a27);
  a13=(a13/a34);
  a34=(a43*a20);
  a27=(a26*a30);
  a34=(a34+a27);
  a34=(a34-a44);
  a41=(a41*a34);
  a43=(a26*a43);
  a30=(a30*a20);
  a43=(a43-a30);
  a43=(a43-a47);
  a32=(a32*a43);
  a41=(a41-a32);
  a13=(a13-a41);
  a41=(a1*a13);
  a41=(a8+a41);
  a0=(a0+a41);
  a0=(a0/a16);
  a0=(a1*a0);
  a0=(a6+a0);
  if (res[0]!=0) res[0][1]=a0;
  a4=(a4*a23);
  a4=(a14+a4);
  a4=(a3*a4);
  a4=(a14+a4);
  a0=(a28*a29);
  a0=(a14+a0);
  a0=(a3*a0);
  a4=(a4+a0);
  a39=(a25*a39);
  a34=(a34-a13);
  a36=(a36*a34);
  a19=(a19*a43);
  a36=(a36+a19);
  a36=(a36-a35);
  a36=(a21*a36);
  a36=(a36+a13);
  a39=(a39-a36);
  a36=(a1*a39);
  a36=(a14+a36);
  a4=(a4+a36);
  a4=(a4/a16);
  a4=(a1*a4);
  a4=(a12+a4);
  if (res[0]!=0) res[0][2]=a4;
  a9=(a3*a9);
  a11=(a11+a9);
  a9=(a3*a20);
  a11=(a11+a9);
  a9=(a28*a37);
  a9=(a8+a9);
  a9=(a1*a9);
  a6=(a6+a9);
  a9=cos(a6);
  a4=(a1*a13);
  a4=(a8+a4);
  a6=sin(a6);
  a20=(a1*a20);
  a20=(a2+a20);
  a36=(a6*a20);
  a35=(a4*a36);
  a28=(a28*a29);
  a28=(a14+a28);
  a28=(a1*a28);
  a12=(a12+a28);
  a28=cos(a12);
  a19=(a1*a39);
  a19=(a14+a19);
  a43=(a19+a4);
  a12=sin(a12);
  a20=(a9*a20);
  a34=(a20-a4);
  a0=(a12*a34);
  a41=(a28*a36);
  a0=(a0+a41);
  a41=(a43*a0);
  a32=(a0*a43);
  a32=(a22+a32);
  a47=(a21*a32);
  a0=(a19*a0);
  a30=(a25*a0);
  a47=(a47-a30);
  a41=(a41+a47);
  a47=(a28*a41);
  a34=(a28*a34);
  a30=(a12*a36);
  a34=(a34-a30);
  a30=(a34-a43);
  a43=(a43*a30);
  a19=(a19*a34);
  a43=(a43-a19);
  a19=(a12*a43);
  a47=(a47-a19);
  a35=(a35+a47);
  a19=(a25*a28);
  a34=(a19*a28);
  a30=casadi_sq(a12);
  a34=(a34+a30);
  a18=(a18-a34);
  a30=(a3+a34);
  a44=(a18/a30);
  a22=(a22-a47);
  a47=(a36*a4);
  a22=(a22-a47);
  a31=(a31-a22);
  a22=(a44*a31);
  a34=(a33+a34);
  a47=(a44*a18);
  a34=(a34-a47);
  a36=(a4*a36);
  a47=(a34*a36);
  a19=(a19*a12);
  a27=(a12*a28);
  a19=(a19-a27);
  a27=(a44*a19);
  a27=(a19+a27);
  a40=(a4*a20);
  a46=(a27*a40);
  a47=(a47+a46);
  a22=(a22-a47);
  a35=(a35+a22);
  a35=(a9*a35);
  a20=(a20-a4);
  a4=(a4*a20);
  a41=(a12*a41);
  a43=(a28*a43);
  a41=(a41+a43);
  a4=(a4+a41);
  a41=(a25*a12);
  a43=(a41*a28);
  a20=(a28*a12);
  a43=(a43-a20);
  a20=(a19/a30);
  a18=(a20*a18);
  a43=(a43+a18);
  a18=(a43*a36);
  a41=(a41*a12);
  a22=casadi_sq(a28);
  a41=(a41+a22);
  a33=(a33+a41);
  a19=(a20*a19);
  a33=(a33-a19);
  a19=(a33*a40);
  a18=(a18+a19);
  a19=(a20*a31);
  a18=(a18+a19);
  a4=(a4-a18);
  a4=(a6*a4);
  a35=(a35-a4);
  a5=(a5-a35);
  a35=(a9*a34);
  a4=(a6*a43);
  a35=(a35-a4);
  a35=(a35*a9);
  a4=(a9*a27);
  a18=(a6*a33);
  a4=(a4-a18);
  a4=(a4*a6);
  a35=(a35-a4);
  a17=(a17+a35);
  a5=(a5/a17);
  a34=(a6*a34);
  a43=(a9*a43);
  a34=(a34+a43);
  a34=(a34*a9);
  a27=(a6*a27);
  a33=(a9*a33);
  a27=(a27+a33);
  a27=(a27*a6);
  a34=(a34-a27);
  a34=(a34/a17);
  a34=(a26*a34);
  a5=(a5-a34);
  a11=(a11+a5);
  a11=(a11/a16);
  a11=(a1*a11);
  a2=(a2+a11);
  if (res[1]!=0) res[1][0]=a2;
  a37=(a3*a37);
  a24=(a24+a37);
  a13=(a3*a13);
  a24=(a24+a13);
  a31=(a31/a30);
  a30=(a9*a5);
  a13=(a26*a6);
  a30=(a30+a13);
  a30=(a30-a36);
  a44=(a44*a30);
  a26=(a26*a9);
  a6=(a6*a5);
  a26=(a26-a6);
  a26=(a26-a40);
  a20=(a20*a26);
  a44=(a44-a20);
  a31=(a31-a44);
  a24=(a24+a31);
  a24=(a24/a16);
  a24=(a1*a24);
  a8=(a8+a24);
  if (res[1]!=0) res[1][1]=a8;
  a29=(a3*a29);
  a23=(a23+a29);
  a3=(a3*a39);
  a23=(a23+a3);
  a25=(a25*a32);
  a30=(a30-a31);
  a28=(a28*a30);
  a12=(a12*a26);
  a28=(a28+a12);
  a28=(a28-a0);
  a21=(a21*a28);
  a21=(a21+a31);
  a25=(a25-a21);
  a23=(a23+a25);
  a23=(a23/a16);
  a1=(a1*a23);
  a14=(a14+a1);
  if (res[1]!=0) res[1][2]=a14;
  return 0;
}

CASADI_SYMBOL_EXPORT int eval_forward_dynamics(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int eval_forward_dynamics_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int eval_forward_dynamics_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void eval_forward_dynamics_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int eval_forward_dynamics_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void eval_forward_dynamics_release(int mem) {
}

CASADI_SYMBOL_EXPORT void eval_forward_dynamics_incref(void) {
}

CASADI_SYMBOL_EXPORT void eval_forward_dynamics_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int eval_forward_dynamics_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int eval_forward_dynamics_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real eval_forward_dynamics_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* eval_forward_dynamics_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* eval_forward_dynamics_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* eval_forward_dynamics_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s0;
    case 3: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* eval_forward_dynamics_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int eval_forward_dynamics_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
