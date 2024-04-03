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
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a6, a7, a8, a9;
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
  a21=-4.0000000000000002e-01;
  a22=arg[2]? arg[2][2] : 0;
  a23=5.0000000000000000e-01;
  a24=(a23*a15);
  a25=(a19*a24);
  a25=(a22+a25);
  a26=(a21*a25);
  a27=8.0000000000000004e-01;
  a19=(a14*a19);
  a28=(a27*a19);
  a26=(a26-a28);
  a20=(a20+a26);
  a26=(a13*a20);
  a18=(a13*a18);
  a28=(a16*a10);
  a18=(a18-a28);
  a24=(a18-a24);
  a15=(a15*a24);
  a18=(a14*a18);
  a15=(a15-a18);
  a18=(a16*a15);
  a26=(a26-a18);
  a11=(a11+a26);
  a18=-5.0000000000000000e-01;
  a24=(a27*a13);
  a28=(a24*a13);
  a29=casadi_sq(a16);
  a28=(a28+a29);
  a29=(a18-a28);
  a30=1.2500000000000000e+00;
  a31=(a30+a28);
  a32=(a29/a31);
  a33=arg[2]? arg[2][1] : 0;
  a26=(a22-a26);
  a34=(a23*a8);
  a35=(a10*a34);
  a26=(a26-a35);
  a26=(a33-a26);
  a35=(a32*a26);
  a36=1.;
  a28=(a36+a28);
  a37=(a32*a29);
  a28=(a28-a37);
  a10=(a8*a10);
  a37=(a28*a10);
  a24=(a24*a16);
  a38=(a16*a13);
  a24=(a24-a38);
  a38=(a32*a24);
  a38=(a24+a38);
  a39=(a8*a17);
  a40=(a38*a39);
  a37=(a37+a40);
  a35=(a35-a37);
  a11=(a11+a35);
  a11=(a7*a11);
  a17=(a17-a34);
  a17=(a8*a17);
  a20=(a16*a20);
  a15=(a13*a15);
  a20=(a20+a15);
  a17=(a17+a20);
  a20=(a27*a16);
  a15=(a20*a13);
  a34=(a13*a16);
  a15=(a15-a34);
  a34=(a24/a31);
  a29=(a34*a29);
  a15=(a15+a29);
  a29=(a15*a10);
  a20=(a20*a16);
  a35=casadi_sq(a13);
  a20=(a20+a35);
  a20=(a36+a20);
  a24=(a34*a24);
  a20=(a20-a24);
  a24=(a20*a39);
  a29=(a29+a24);
  a24=(a34*a26);
  a29=(a29+a24);
  a17=(a17-a29);
  a17=(a9*a17);
  a11=(a11-a17);
  a11=(a5-a11);
  a17=5.;
  a29=(a7*a28);
  a24=(a9*a15);
  a29=(a29-a24);
  a29=(a29*a7);
  a24=(a7*a38);
  a35=(a9*a20);
  a24=(a24-a35);
  a24=(a24*a9);
  a29=(a29-a24);
  a29=(a17+a29);
  a11=(a11/a29);
  a24=9.8100000000000005e+00;
  a28=(a9*a28);
  a15=(a7*a15);
  a28=(a28+a15);
  a28=(a28*a7);
  a38=(a9*a38);
  a20=(a7*a20);
  a38=(a38+a20);
  a38=(a38*a9);
  a28=(a28-a38);
  a28=(a28/a29);
  a28=(a24*a28);
  a11=(a11-a28);
  a28=(a4*a11);
  a28=(a2+a28);
  a28=(a3*a28);
  a28=(a2+a28);
  a29=(a1/a3);
  a38=(a1/a3);
  a20=(a38*a8);
  a20=(a6+a20);
  a15=cos(a20);
  a35=(a1/a3);
  a26=(a26/a31);
  a31=(a7*a11);
  a37=(a24*a9);
  a31=(a31+a37);
  a31=(a31-a10);
  a32=(a32*a31);
  a7=(a24*a7);
  a9=(a9*a11);
  a7=(a7-a9);
  a7=(a7-a39);
  a34=(a34*a7);
  a32=(a32-a34);
  a26=(a26-a32);
  a32=(a35*a26);
  a32=(a8+a32);
  a20=sin(a20);
  a34=(a35*a11);
  a34=(a2+a34);
  a39=(a20*a34);
  a9=(a32*a39);
  a38=(a38*a14);
  a38=(a12+a38);
  a10=cos(a38);
  a25=(a27*a25);
  a31=(a31-a26);
  a13=(a13*a31);
  a16=(a16*a7);
  a13=(a13+a16);
  a13=(a13-a19);
  a13=(a21*a13);
  a13=(a13+a26);
  a25=(a25-a13);
  a35=(a35*a25);
  a35=(a14+a35);
  a13=(a35+a32);
  a38=sin(a38);
  a34=(a15*a34);
  a19=(a34-a32);
  a16=(a38*a19);
  a7=(a10*a39);
  a16=(a16+a7);
  a7=(a13*a16);
  a31=(a23*a13);
  a37=(a16*a31);
  a37=(a22+a37);
  a40=(a21*a37);
  a16=(a35*a16);
  a41=(a27*a16);
  a40=(a40-a41);
  a7=(a7+a40);
  a40=(a10*a7);
  a19=(a10*a19);
  a41=(a38*a39);
  a19=(a19-a41);
  a31=(a19-a31);
  a13=(a13*a31);
  a35=(a35*a19);
  a13=(a13-a35);
  a35=(a38*a13);
  a40=(a40-a35);
  a9=(a9+a40);
  a35=(a27*a10);
  a19=(a35*a10);
  a31=casadi_sq(a38);
  a19=(a19+a31);
  a31=(a18-a19);
  a41=(a30+a19);
  a42=(a31/a41);
  a40=(a22-a40);
  a43=(a23*a32);
  a44=(a39*a43);
  a40=(a40-a44);
  a40=(a33-a40);
  a44=(a42*a40);
  a19=(a36+a19);
  a45=(a42*a31);
  a19=(a19-a45);
  a39=(a32*a39);
  a45=(a19*a39);
  a35=(a35*a38);
  a46=(a38*a10);
  a35=(a35-a46);
  a46=(a42*a35);
  a46=(a35+a46);
  a47=(a32*a34);
  a48=(a46*a47);
  a45=(a45+a48);
  a44=(a44-a45);
  a9=(a9+a44);
  a9=(a15*a9);
  a34=(a34-a43);
  a32=(a32*a34);
  a7=(a38*a7);
  a13=(a10*a13);
  a7=(a7+a13);
  a32=(a32+a7);
  a7=(a27*a38);
  a13=(a7*a10);
  a34=(a10*a38);
  a13=(a13-a34);
  a34=(a35/a41);
  a31=(a34*a31);
  a13=(a13+a31);
  a31=(a13*a39);
  a7=(a7*a38);
  a43=casadi_sq(a10);
  a7=(a7+a43);
  a7=(a36+a7);
  a35=(a34*a35);
  a7=(a7-a35);
  a35=(a7*a47);
  a31=(a31+a35);
  a35=(a34*a40);
  a31=(a31+a35);
  a32=(a32-a31);
  a32=(a20*a32);
  a9=(a9-a32);
  a9=(a5-a9);
  a32=(a15*a19);
  a31=(a20*a13);
  a32=(a32-a31);
  a32=(a32*a15);
  a31=(a15*a46);
  a35=(a20*a7);
  a31=(a31-a35);
  a31=(a31*a20);
  a32=(a32-a31);
  a32=(a17+a32);
  a9=(a9/a32);
  a19=(a20*a19);
  a13=(a15*a13);
  a19=(a19+a13);
  a19=(a19*a15);
  a46=(a20*a46);
  a7=(a15*a7);
  a46=(a46+a7);
  a46=(a46*a20);
  a19=(a19-a46);
  a19=(a19/a32);
  a19=(a24*a19);
  a9=(a9-a19);
  a19=(a29*a9);
  a19=(a2+a19);
  a19=(a3*a19);
  a28=(a28+a19);
  a19=(a1/a3);
  a32=(a4*a26);
  a32=(a8+a32);
  a32=(a19*a32);
  a32=(a6+a32);
  a46=cos(a32);
  a7=(a1/a3);
  a40=(a40/a41);
  a41=(a15*a9);
  a13=(a24*a20);
  a41=(a41+a13);
  a41=(a41-a39);
  a42=(a42*a41);
  a15=(a24*a15);
  a20=(a20*a9);
  a15=(a15-a20);
  a15=(a15-a47);
  a34=(a34*a15);
  a42=(a42-a34);
  a40=(a40-a42);
  a42=(a7*a40);
  a42=(a8+a42);
  a32=sin(a32);
  a34=(a7*a9);
  a34=(a2+a34);
  a47=(a32*a34);
  a20=(a42*a47);
  a39=(a4*a25);
  a39=(a14+a39);
  a19=(a19*a39);
  a19=(a12+a19);
  a39=cos(a19);
  a37=(a27*a37);
  a41=(a41-a40);
  a10=(a10*a41);
  a38=(a38*a15);
  a10=(a10+a38);
  a10=(a10-a16);
  a10=(a21*a10);
  a10=(a10+a40);
  a37=(a37-a10);
  a7=(a7*a37);
  a7=(a14+a7);
  a10=(a7+a42);
  a19=sin(a19);
  a34=(a46*a34);
  a16=(a34-a42);
  a38=(a19*a16);
  a15=(a39*a47);
  a38=(a38+a15);
  a15=(a10*a38);
  a41=(a23*a10);
  a13=(a38*a41);
  a13=(a22+a13);
  a31=(a21*a13);
  a38=(a7*a38);
  a35=(a27*a38);
  a31=(a31-a35);
  a15=(a15+a31);
  a31=(a39*a15);
  a16=(a39*a16);
  a35=(a19*a47);
  a16=(a16-a35);
  a41=(a16-a41);
  a10=(a10*a41);
  a7=(a7*a16);
  a10=(a10-a7);
  a7=(a19*a10);
  a31=(a31-a7);
  a20=(a20+a31);
  a7=(a27*a39);
  a16=(a7*a39);
  a41=casadi_sq(a19);
  a16=(a16+a41);
  a41=(a18-a16);
  a35=(a30+a16);
  a43=(a41/a35);
  a31=(a22-a31);
  a44=(a23*a42);
  a45=(a47*a44);
  a31=(a31-a45);
  a31=(a33-a31);
  a45=(a43*a31);
  a16=(a36+a16);
  a48=(a43*a41);
  a16=(a16-a48);
  a47=(a42*a47);
  a48=(a16*a47);
  a7=(a7*a19);
  a49=(a19*a39);
  a7=(a7-a49);
  a49=(a43*a7);
  a49=(a7+a49);
  a50=(a42*a34);
  a51=(a49*a50);
  a48=(a48+a51);
  a45=(a45-a48);
  a20=(a20+a45);
  a20=(a46*a20);
  a34=(a34-a44);
  a42=(a42*a34);
  a15=(a19*a15);
  a10=(a39*a10);
  a15=(a15+a10);
  a42=(a42+a15);
  a15=(a27*a19);
  a10=(a15*a39);
  a34=(a39*a19);
  a10=(a10-a34);
  a34=(a7/a35);
  a41=(a34*a41);
  a10=(a10+a41);
  a41=(a10*a47);
  a15=(a15*a19);
  a44=casadi_sq(a39);
  a15=(a15+a44);
  a15=(a36+a15);
  a7=(a34*a7);
  a15=(a15-a7);
  a7=(a15*a50);
  a41=(a41+a7);
  a7=(a34*a31);
  a41=(a41+a7);
  a42=(a42-a41);
  a42=(a32*a42);
  a20=(a20-a42);
  a20=(a5-a20);
  a42=(a46*a16);
  a41=(a32*a10);
  a42=(a42-a41);
  a42=(a42*a46);
  a41=(a46*a49);
  a7=(a32*a15);
  a41=(a41-a7);
  a41=(a41*a32);
  a42=(a42-a41);
  a42=(a17+a42);
  a20=(a20/a42);
  a16=(a32*a16);
  a10=(a46*a10);
  a16=(a16+a10);
  a16=(a16*a46);
  a49=(a32*a49);
  a15=(a46*a15);
  a49=(a49+a15);
  a49=(a49*a32);
  a16=(a16-a49);
  a16=(a16/a42);
  a16=(a24*a16);
  a20=(a20-a16);
  a16=(a1*a20);
  a16=(a2+a16);
  a28=(a28+a16);
  a16=6.;
  a28=(a28/a16);
  a28=(a1*a28);
  a0=(a0+a28);
  if (res[0]!=0) res[0][0]=a0;
  a0=(a4*a26);
  a0=(a8+a0);
  a0=(a3*a0);
  a0=(a8+a0);
  a28=(a29*a40);
  a28=(a8+a28);
  a28=(a3*a28);
  a0=(a0+a28);
  a31=(a31/a35);
  a35=(a46*a20);
  a28=(a24*a32);
  a35=(a35+a28);
  a35=(a35-a47);
  a43=(a43*a35);
  a46=(a24*a46);
  a32=(a32*a20);
  a46=(a46-a32);
  a46=(a46-a50);
  a34=(a34*a46);
  a43=(a43-a34);
  a31=(a31-a43);
  a43=(a1*a31);
  a43=(a8+a43);
  a0=(a0+a43);
  a0=(a0/a16);
  a0=(a1*a0);
  a0=(a6+a0);
  if (res[0]!=0) res[0][1]=a0;
  a4=(a4*a25);
  a4=(a14+a4);
  a4=(a3*a4);
  a4=(a14+a4);
  a0=(a29*a37);
  a0=(a14+a0);
  a0=(a3*a0);
  a4=(a4+a0);
  a13=(a27*a13);
  a35=(a35-a31);
  a39=(a39*a35);
  a19=(a19*a46);
  a39=(a39+a19);
  a39=(a39-a38);
  a39=(a21*a39);
  a39=(a39+a31);
  a13=(a13-a39);
  a39=(a1*a13);
  a39=(a14+a39);
  a4=(a4+a39);
  a4=(a4/a16);
  a4=(a1*a4);
  a4=(a12+a4);
  if (res[0]!=0) res[0][2]=a4;
  a9=(a3*a9);
  a11=(a11+a9);
  a9=(a3*a20);
  a11=(a11+a9);
  a9=(a29*a40);
  a9=(a8+a9);
  a9=(a1*a9);
  a6=(a6+a9);
  a9=cos(a6);
  a4=(a1*a31);
  a4=(a8+a4);
  a6=sin(a6);
  a20=(a1*a20);
  a20=(a2+a20);
  a39=(a6*a20);
  a38=(a4*a39);
  a29=(a29*a37);
  a29=(a14+a29);
  a29=(a1*a29);
  a12=(a12+a29);
  a29=cos(a12);
  a19=(a1*a13);
  a19=(a14+a19);
  a46=(a19+a4);
  a12=sin(a12);
  a20=(a9*a20);
  a35=(a20-a4);
  a0=(a12*a35);
  a43=(a29*a39);
  a0=(a0+a43);
  a43=(a46*a0);
  a34=(a23*a46);
  a50=(a0*a34);
  a50=(a22+a50);
  a32=(a21*a50);
  a0=(a19*a0);
  a47=(a27*a0);
  a32=(a32-a47);
  a43=(a43+a32);
  a32=(a29*a43);
  a35=(a29*a35);
  a47=(a12*a39);
  a35=(a35-a47);
  a34=(a35-a34);
  a46=(a46*a34);
  a19=(a19*a35);
  a46=(a46-a19);
  a19=(a12*a46);
  a32=(a32-a19);
  a38=(a38+a32);
  a19=(a27*a29);
  a35=(a19*a29);
  a34=casadi_sq(a12);
  a35=(a35+a34);
  a18=(a18-a35);
  a30=(a30+a35);
  a34=(a18/a30);
  a22=(a22-a32);
  a23=(a23*a4);
  a32=(a39*a23);
  a22=(a22-a32);
  a33=(a33-a22);
  a22=(a34*a33);
  a35=(a36+a35);
  a32=(a34*a18);
  a35=(a35-a32);
  a39=(a4*a39);
  a32=(a35*a39);
  a19=(a19*a12);
  a47=(a12*a29);
  a19=(a19-a47);
  a47=(a34*a19);
  a47=(a19+a47);
  a28=(a4*a20);
  a42=(a47*a28);
  a32=(a32+a42);
  a22=(a22-a32);
  a38=(a38+a22);
  a38=(a9*a38);
  a20=(a20-a23);
  a4=(a4*a20);
  a43=(a12*a43);
  a46=(a29*a46);
  a43=(a43+a46);
  a4=(a4+a43);
  a43=(a27*a12);
  a46=(a43*a29);
  a20=(a29*a12);
  a46=(a46-a20);
  a20=(a19/a30);
  a18=(a20*a18);
  a46=(a46+a18);
  a18=(a46*a39);
  a43=(a43*a12);
  a23=casadi_sq(a29);
  a43=(a43+a23);
  a36=(a36+a43);
  a19=(a20*a19);
  a36=(a36-a19);
  a19=(a36*a28);
  a18=(a18+a19);
  a19=(a20*a33);
  a18=(a18+a19);
  a4=(a4-a18);
  a4=(a6*a4);
  a38=(a38-a4);
  a5=(a5-a38);
  a38=(a9*a35);
  a4=(a6*a46);
  a38=(a38-a4);
  a38=(a38*a9);
  a4=(a9*a47);
  a18=(a6*a36);
  a4=(a4-a18);
  a4=(a4*a6);
  a38=(a38-a4);
  a17=(a17+a38);
  a5=(a5/a17);
  a35=(a6*a35);
  a46=(a9*a46);
  a35=(a35+a46);
  a35=(a35*a9);
  a47=(a6*a47);
  a36=(a9*a36);
  a47=(a47+a36);
  a47=(a47*a6);
  a35=(a35-a47);
  a35=(a35/a17);
  a35=(a24*a35);
  a5=(a5-a35);
  a11=(a11+a5);
  a11=(a11/a16);
  a11=(a1*a11);
  a2=(a2+a11);
  if (res[1]!=0) res[1][0]=a2;
  a40=(a3*a40);
  a26=(a26+a40);
  a31=(a3*a31);
  a26=(a26+a31);
  a33=(a33/a30);
  a30=(a9*a5);
  a31=(a24*a6);
  a30=(a30+a31);
  a30=(a30-a39);
  a34=(a34*a30);
  a24=(a24*a9);
  a6=(a6*a5);
  a24=(a24-a6);
  a24=(a24-a28);
  a20=(a20*a24);
  a34=(a34-a20);
  a33=(a33-a34);
  a26=(a26+a33);
  a26=(a26/a16);
  a26=(a1*a26);
  a8=(a8+a26);
  if (res[1]!=0) res[1][1]=a8;
  a37=(a3*a37);
  a25=(a25+a37);
  a3=(a3*a13);
  a25=(a25+a3);
  a27=(a27*a50);
  a30=(a30-a33);
  a29=(a29*a30);
  a12=(a12*a24);
  a29=(a29+a12);
  a29=(a29-a0);
  a21=(a21*a29);
  a21=(a21+a33);
  a27=(a27-a21);
  a25=(a25+a27);
  a25=(a25/a16);
  a1=(a1*a25);
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