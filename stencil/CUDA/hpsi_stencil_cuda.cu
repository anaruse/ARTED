#include <stdio.h>
#include <assert.h>
#include <cuComplex.h>

#define USE_V1  // naive version, each thread handles one cell
// #define USE_V2  // each thread handles multiple cells along x-dir. keep data of 9 cells along x-dir on registers to reduce memory traffic.
// #define USE_V4  // 2 kernels. 1st kernel does x-dir stencil computes. 2nd kernel does y-dir and z-dir stencil computes.
// #define USE_V5  // 2 kernels, shared memory is used in 2nd kernel (this is good in case of small grid size such as 16x16x16)
// #define USE_V8  // 2 kernels, 1st kernel is optimized for reducing memory access as much as possible.
// #define USE_CPU

#define USE_CONST  // use constant memory for arrays such as C() and D().

__host__ __device__ inline int CALC_INDEX2( int x, int nx, int y, int ny )
{
    return( x + nx * y );
}

__host__ __device__ inline int CALC_INDEX3( int x, int nx, int y, int ny, int z, int nz )
{
    return( CALC_INDEX2( CALC_INDEX2(x,nx,y,ny), nx*ny, z, nz ) );
}

__host__ __device__ inline int CALC_INDEX4( int x, int nx, int y, int ny, int z, int nz, int w, int nw )
{
    return( CALC_INDEX2( CALC_INDEX3(x,nx,y,ny,z,nz), nx*ny*nz, w, nw ) );
}

__host__ __device__ inline cuDoubleComplex cuCmul( double x, cuDoubleComplex y )
{
    cuDoubleComplex val;
    val = make_cuDoubleComplex ( x * cuCreal(y), x * cuCimag(y) );
    return val;
}

__host__ __device__ inline cuDoubleComplex cuCmulI( cuDoubleComplex x )
{
    cuDoubleComplex val;
    val = make_cuDoubleComplex ( - cuCimag(x), cuCreal(x) );
    return val;
}

#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
			     fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
			     exit(-1);}} while(0)
#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define DIV_CEIL(X,Y) (((X)+(Y)-1)/(Y))


/* */

#ifdef USE_CONST
#define MAX_NKB 512
__constant__ double  _C_const[12];
__constant__ double  _D_const[12*MAX_NKB];
#endif

/* */

extern __shared__ void* _dyn_smem[];

/************************************************************/

#ifdef USE_V1

#define A(KB)       _A[(KB)]
#define B(Z,Y,X)    _B[CALC_INDEX3((Z),NLz,(Y),NLy,(X),NLx)]
#ifdef USE_CONST
#define C(I)        _C_const[((I)-1)]
#define D(I)        _D_const[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#else
#define C(I)        _C[((I)-1)]
#define D(I)        _D[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#endif
#define E(Z,Y,X)    _E[CALC_INDEX3((Z),PNLz,(Y),PNLy,(X),PNLx)]
#define F(Z,Y,X,KB) _F[CALC_INDEX4((Z),PNLz,(Y),PNLy,(X),PNLx,(KB),Nkb)]

#define E_IDX(DT)  _E[CALC_INDEX3(iz,PNLz,iy,PNLy,modx[ix+(DT)+NLx],PNLx)]
#define E_IDY(DT)  _E[CALC_INDEX3(iz,PNLz,mody[iy+(DT)+NLy],PNLy,ix,PNLx)]
#define E_IDZ(DT)  _E[CALC_INDEX3(modz[iz+(DT)+NLz],PNLz,iy,PNLy,ix,PNLx)]

/* */
__global__ void hpsi1_rt_stencil_kern_v1(int Nkb,
					 const double * __restrict__ _A, const double * __restrict__ _B, const double * __restrict__ _C, const double * __restrict__ _D,
					 const cuDoubleComplex * __restrict__ __E, cuDoubleComplex *_F,
					 int PNLx, int PNLy, int PNLz,
					 int NLx, int NLy, int NLz,
					 const int * __restrict__ modx, const int * __restrict__ mody, const int * __restrict__ modz )
{
    int ikb, iyz,iz,iy,ix;
    cuDoubleComplex v,w, val;

    ikb = blockIdx.y;
    if ( ikb >= Nkb ) return;

    const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

    iyz = threadIdx.x + blockDim.x * blockIdx.x;
    iz = iyz % NLz;
    iy = iyz / NLz;
    if ( iy >= NLy ) return;

    for (ix=0; ix < NLx; ix++) {

	v =         cuCmul(C( 9),(cuCadd(E_IDZ(1),E_IDZ(-1))));
	v = cuCadd( cuCmul(C(10),(cuCadd(E_IDZ(2),E_IDZ(-2)))), v );
	v = cuCadd( cuCmul(C(11),(cuCadd(E_IDZ(3),E_IDZ(-3)))), v );
	v = cuCadd( cuCmul(C(12),(cuCadd(E_IDZ(4),E_IDZ(-4)))), v );

	w =         cuCmul(D( 9),(cuCsub(E_IDZ(1),E_IDZ(-1))));
	w = cuCadd( cuCmul(D(10),(cuCsub(E_IDZ(2),E_IDZ(-2)))), w );
	w = cuCadd( cuCmul(D(11),(cuCsub(E_IDZ(3),E_IDZ(-3)))), w );
	w = cuCadd( cuCmul(D(12),(cuCsub(E_IDZ(4),E_IDZ(-4)))), w );

	v = cuCadd( cuCmul(C( 5),(cuCadd(E_IDY(1),E_IDY(-1)))), v );
	v = cuCadd( cuCmul(C( 6),(cuCadd(E_IDY(2),E_IDY(-2)))), v );
	v = cuCadd( cuCmul(C( 7),(cuCadd(E_IDY(3),E_IDY(-3)))), v );
	v = cuCadd( cuCmul(C( 8),(cuCadd(E_IDY(4),E_IDY(-4)))), v );

	w = cuCadd( cuCmul(D( 5),(cuCsub(E_IDY(1),E_IDY(-1)))), w );
	w = cuCadd( cuCmul(D( 6),(cuCsub(E_IDY(2),E_IDY(-2)))), w );
	w = cuCadd( cuCmul(D( 7),(cuCsub(E_IDY(3),E_IDY(-3)))), w );
	w = cuCadd( cuCmul(D( 8),(cuCsub(E_IDY(4),E_IDY(-4)))), w );

	v = cuCadd( cuCmul(C( 1),(cuCadd(E_IDX(1),E_IDX(-1)))), v );
	v = cuCadd( cuCmul(C( 2),(cuCadd(E_IDX(2),E_IDX(-2)))), v );
	v = cuCadd( cuCmul(C( 3),(cuCadd(E_IDX(3),E_IDX(-3)))), v );
	v = cuCadd( cuCmul(C( 4),(cuCadd(E_IDX(4),E_IDX(-4)))), v );

	w = cuCadd( cuCmul(D( 1),(cuCsub(E_IDX(1),E_IDX(-1)))), w );
	w = cuCadd( cuCmul(D( 2),(cuCsub(E_IDX(2),E_IDX(-2)))), w );
	w = cuCadd( cuCmul(D( 3),(cuCsub(E_IDX(3),E_IDX(-3)))), w );
	w = cuCadd( cuCmul(D( 4),(cuCsub(E_IDX(4),E_IDX(-4)))), w );

	val = cuCmul((B(iz,iy,ix)+A(ikb)), E(iz,iy,ix));
	val = cuCsub( val, cuCmul(0.5, v) );
	val = make_cuDoubleComplex( cuCreal(val) + cuCimag(w), cuCimag(val) - cuCreal(w) );

	F(iz,iy,ix,ikb) = val;
    }
}
#endif

/************************************************************/

#ifdef USE_V2

#define A(KB)       _A[(KB)]
#define B(Z,Y,X)    _B[CALC_INDEX3((Z),NLz,(Y),NLy,(X),NLx)]
#ifdef USE_CONST
#define C(I)        _C_const[((I)-1)]
#define D(I)        _D_const[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#else
#define C(I)        _C[((I)-1)]
#define D(I)        _D[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#endif
#define E(Z,Y,X)    _E[CALC_INDEX3((Z),PNLz,(Y),PNLy,(X),PNLx)]
#define F(Z,Y,X,KB) _F[CALC_INDEX4((Z),PNLz,(Y),PNLy,(X),PNLx,(KB),Nkb)]

#define E_IDX(DT)  _E[CALC_INDEX3(iz,PNLz,iy,PNLy,modx[ix+(DT)+NLx],PNLx)]
#define E_IDY(DT)  _E[CALC_INDEX3(iz,PNLz,mody[iy+(DT)+NLy],PNLy,ix,PNLx)]
#define E_IDZ(DT)  _E[CALC_INDEX3(modz[iz+(DT)+NLz],PNLz,iy,PNLy,ix,PNLx)]

// __launch_bounds__(128,4)
__global__ void hpsi1_rt_stencil_kern_v2(int Nkb,
					 const double * __restrict__ _A, const double * __restrict__ _B, const double * __restrict__ _C, const double * __restrict__ _D,
					 const cuDoubleComplex * __restrict__ __E, cuDoubleComplex *_F,
					 int PNLx, int PNLy, int PNLz,
					 int NLx, int NLy, int NLz,
					 const int * __restrict__ modx, const int * __restrict__ mody, const int * __restrict__ modz )
{
    int ikb, iyz,iz,iy,ix;
    cuDoubleComplex v,w, val;

    ikb = blockIdx.y;
    if ( ikb >= Nkb ) return;

    const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

    iyz = threadIdx.x + blockDim.x * blockIdx.x;
    iz = iyz % NLz;
    iy = iyz / NLz;
    if ( iy >= NLy ) return;

    cuDoubleComplex E_idx[9];

    ix = 0;
    E_idx[0] = E_IDX(-4);
    E_idx[1] = E_IDX(-3);
    E_idx[2] = E_IDX(-2);
    E_idx[3] = E_IDX(-1);
    E_idx[4] = E_IDX( 0);
    E_idx[5] = E_IDX( 1);
    E_idx[6] = E_IDX( 2);
    E_idx[7] = E_IDX( 3);

    for (ix=0; ix < NLx; ix++) {
	E_idx[8] = E_IDX( 4);

	v =         cuCmul(C(12),(cuCadd(E_IDZ(4),E_IDZ(-4))));
	w =         cuCmul(D(12),(cuCsub(E_IDZ(4),E_IDZ(-4))));
	v = cuCadd( cuCmul(C(11),(cuCadd(E_IDZ(3),E_IDZ(-3)))), v );
	w = cuCadd( cuCmul(D(11),(cuCsub(E_IDZ(3),E_IDZ(-3)))), w );
	v = cuCadd( cuCmul(C(10),(cuCadd(E_IDZ(2),E_IDZ(-2)))), v );
	w = cuCadd( cuCmul(D(10),(cuCsub(E_IDZ(2),E_IDZ(-2)))), w );
	v = cuCadd( cuCmul(C( 9),(cuCadd(E_IDZ(1),E_IDZ(-1)))), v );
	w = cuCadd( cuCmul(D( 9),(cuCsub(E_IDZ(1),E_IDZ(-1)))), w );

	v = cuCadd( cuCmul(C( 8),(cuCadd(E_IDY(4),E_IDY(-4)))), v );
	w = cuCadd( cuCmul(D( 8),(cuCsub(E_IDY(4),E_IDY(-4)))), w );
	v = cuCadd( cuCmul(C( 7),(cuCadd(E_IDY(3),E_IDY(-3)))), v );
	w = cuCadd( cuCmul(D( 7),(cuCsub(E_IDY(3),E_IDY(-3)))), w );
	v = cuCadd( cuCmul(C( 6),(cuCadd(E_IDY(2),E_IDY(-2)))), v );
	w = cuCadd( cuCmul(D( 6),(cuCsub(E_IDY(2),E_IDY(-2)))), w );
	v = cuCadd( cuCmul(C( 5),(cuCadd(E_IDY(1),E_IDY(-1)))), v );
	w = cuCadd( cuCmul(D( 5),(cuCsub(E_IDY(1),E_IDY(-1)))), w );

	v = cuCadd( cuCmul(C( 1),(cuCadd(E_idx[5],E_idx[3]))), v );
	w = cuCadd( cuCmul(D( 1),(cuCsub(E_idx[5],E_idx[3]))), w );
	v = cuCadd( cuCmul(C( 2),(cuCadd(E_idx[6],E_idx[2]))), v );
	w = cuCadd( cuCmul(D( 2),(cuCsub(E_idx[6],E_idx[2]))), w );
	v = cuCadd( cuCmul(C( 3),(cuCadd(E_idx[7],E_idx[1]))), v );
	w = cuCadd( cuCmul(D( 3),(cuCsub(E_idx[7],E_idx[1]))), w );
	v = cuCadd( cuCmul(C( 4),(cuCadd(E_idx[8],E_idx[0]))), v );
	w = cuCadd( cuCmul(D( 4),(cuCsub(E_idx[8],E_idx[0]))), w );

	val = cuCmul((B(iz,iy,ix)+A(ikb)), E_idx[4]);
	val = cuCsub( val, cuCmul(0.5, v) );
	val = cuCsub( val, cuCmulI(w) );

	F(iz,iy,ix,ikb) = val;

	E_idx[0] = E_idx[1];
	E_idx[1] = E_idx[2];
	E_idx[2] = E_idx[3];
	E_idx[3] = E_idx[4];
	E_idx[4] = E_idx[5];
	E_idx[5] = E_idx[6];
	E_idx[6] = E_idx[7];
	E_idx[7] = E_idx[8];
    }
}
#endif

/************************************************************/

#ifdef USE_V4

#define A(KB)       _A[(KB)]
#define B(Z,Y,X)    _B[CALC_INDEX3((Z),NLz,(Y),NLy,(X),NLx)]
#ifdef USE_CONST
#define C(I)        _C_const[((I)-1)]
#define D(I)        _D_const[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#else
#define C(I)        _C[((I)-1)]
#define D(I)        _D[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#endif
#define E(Z,Y,X)    _E[CALC_INDEX3((Z),PNLz,(Y),PNLy,(X),PNLx)]
#define F(Z,Y,X,KB) _F[CALC_INDEX4((Z),PNLz,(Y),PNLy,(X),PNLx,(KB),Nkb)]

#define E_IDX(DT)  _E[CALC_INDEX3(iz,PNLz,iy,PNLy,modx[ix+(DT)+NLx],PNLx)]
#define E_IDY(DT)  _E[CALC_INDEX3(iz,PNLz,mody[iy+(DT)+NLy],PNLy,ix,PNLx)]
#define E_IDZ(DT)  _E[CALC_INDEX3(modz[iz+(DT)+NLz],PNLz,iy,PNLy,ix,PNLx)]

__launch_bounds__(128,6)
__global__ void hpsi1_rt_stencil_kern_v4_1(int Nkb,
					const double * __restrict__ _A, const double * __restrict__ _B, const double * __restrict__ _C, const double * __restrict__ _D,
					const cuDoubleComplex * __E, cuDoubleComplex *_F,
					   int PNLx, int PNLy, int PNLz,
					int NLx, int NLy, int NLz,
					const int * __restrict__ modx, const int * __restrict__ mody, const int * __restrict__ modz )
{
    int ikb, iyz,iz,iy,ix;
    cuDoubleComplex v,w, val;

    ikb = blockIdx.y;
    if ( ikb >= Nkb ) return;

    const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

    iyz = threadIdx.x + blockDim.x * blockIdx.x;
    iz = iyz % NLz;
    iy = iyz / NLz;
    if ( iy >= NLy ) return;

    cuDoubleComplex E_idx[10];

    ix = 0;
    E_idx[0] = E_IDX(-4);
    E_idx[1] = E_IDX(-3);
    E_idx[2] = E_IDX(-2);
    E_idx[3] = E_IDX(-1);
    E_idx[4] = E_IDX( 0);
    E_idx[5] = E_IDX( 1);
    E_idx[6] = E_IDX( 2);
    E_idx[7] = E_IDX( 3);
    E_idx[8] = E_IDX( 4);

    for (ix=0; ix < NLx; ix++) {
	E_idx[9] = E_IDX( 5);

	v =         cuCmul(C( 1),(cuCadd(E_idx[5],E_idx[3])));
	w =         cuCmul(D( 1),(cuCsub(E_idx[5],E_idx[3])));
	v = cuCadd( cuCmul(C( 2),(cuCadd(E_idx[6],E_idx[2]))), v );
	w = cuCadd( cuCmul(D( 2),(cuCsub(E_idx[6],E_idx[2]))), w );
	v = cuCadd( cuCmul(C( 3),(cuCadd(E_idx[7],E_idx[1]))), v );
	w = cuCadd( cuCmul(D( 3),(cuCsub(E_idx[7],E_idx[1]))), w );
	v = cuCadd( cuCmul(C( 4),(cuCadd(E_idx[8],E_idx[0]))), v );
	w = cuCadd( cuCmul(D( 4),(cuCsub(E_idx[8],E_idx[0]))), w );
	val = cuCmul((B(iz,iy,ix)+A(ikb)), E_idx[4]);
	val = cuCsub( val, cuCmul(0.5, v) );
	val = cuCsub( val, cuCmulI(w) );

	F(iz,iy,ix,ikb) = val;

	E_idx[0] = E_idx[1];
	E_idx[1] = E_idx[2];
	E_idx[2] = E_idx[3];
	E_idx[3] = E_idx[4];
	E_idx[4] = E_idx[5];
	E_idx[5] = E_idx[6];
	E_idx[6] = E_idx[7];
	E_idx[7] = E_idx[8];
	E_idx[8] = E_idx[9];
    }
}

/* */
__launch_bounds__(128,4)
__global__ void hpsi1_rt_stencil_kern_v4_2(int Nkb,
					const double * __restrict__ _A, const double * __restrict__ _B, const double * __restrict__ _C, const double * __restrict__ _D,
					const cuDoubleComplex * __restrict__ __E, cuDoubleComplex *_F,
					   int PNLx, int PNLy, int PNLz,
					int NLx, int NLy, int NLz,
					const int * __restrict__ modx, const int * __restrict__ mody, const int * __restrict__ modz )
{
    int ikb, ixz,iz,iy,ix;
    cuDoubleComplex v,w, val;

    ikb = blockIdx.y;
    if ( ikb >= Nkb ) return;

    const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

    ixz = threadIdx.x + blockDim.x * blockIdx.x;
    iz = ixz % NLz;
    ix = ixz / NLz;
    if ( ix >= NLx ) return;

    cuDoubleComplex E_idy[9];

    iy = 0;
    E_idy[0] = E_IDY(-4);
    E_idy[1] = E_IDY(-3);
    E_idy[2] = E_IDY(-2);
    E_idy[3] = E_IDY(-1);
    E_idy[4] = E_IDY( 0);
    E_idy[5] = E_IDY( 1);
    E_idy[6] = E_IDY( 2);
    E_idy[7] = E_IDY( 3);

    for (iy=0; iy < NLy; iy++) {
	E_idy[8] = E_IDY( 4);

	v =         cuCmul(C( 9),(cuCadd(E_IDZ(1),E_IDZ(-1))));
	w =         cuCmul(D( 9),(cuCsub(E_IDZ(1),E_IDZ(-1))));
	v = cuCadd( cuCmul(C(10),(cuCadd(E_IDZ(2),E_IDZ(-2)))), v );
	w = cuCadd( cuCmul(D(10),(cuCsub(E_IDZ(2),E_IDZ(-2)))), w );
	v = cuCadd( cuCmul(C(11),(cuCadd(E_IDZ(3),E_IDZ(-3)))), v );
	w = cuCadd( cuCmul(D(11),(cuCsub(E_IDZ(3),E_IDZ(-3)))), w );
	v = cuCadd( cuCmul(C(12),(cuCadd(E_IDZ(4),E_IDZ(-4)))), v );
	w = cuCadd( cuCmul(D(12),(cuCsub(E_IDZ(4),E_IDZ(-4)))), w );

	v = cuCadd( cuCmul(C( 5),(cuCadd(E_idy[5],E_idy[3]))), v );
	w = cuCadd( cuCmul(D( 5),(cuCsub(E_idy[5],E_idy[3]))), w );
	v = cuCadd( cuCmul(C( 6),(cuCadd(E_idy[6],E_idy[2]))), v );
	w = cuCadd( cuCmul(D( 6),(cuCsub(E_idy[6],E_idy[2]))), w );
	v = cuCadd( cuCmul(C( 7),(cuCadd(E_idy[7],E_idy[1]))), v );
	w = cuCadd( cuCmul(D( 7),(cuCsub(E_idy[7],E_idy[1]))), w );
	v = cuCadd( cuCmul(C( 8),(cuCadd(E_idy[8],E_idy[0]))), v );
	w = cuCadd( cuCmul(D( 8),(cuCsub(E_idy[8],E_idy[0]))), w );

	val = cuCmul(-0.5, v);
	val = cuCsub( val, cuCmulI(w) );

	F(iz,iy,ix,ikb) = cuCadd(F(iz,iy,ix,ikb), val);

	E_idy[0] = E_idy[1];
	E_idy[1] = E_idy[2];
	E_idy[2] = E_idy[3];
	E_idy[3] = E_idy[4];
	E_idy[4] = E_idy[5];
	E_idy[5] = E_idy[6];
	E_idy[6] = E_idy[7];
	E_idy[7] = E_idy[8];
    }
}
#endif

/************************************************************/

#ifdef USE_V5

#define A(KB)       _A[(KB)]
#define B(Z,Y,X)    _B[CALC_INDEX3((Z),NLz,(Y),NLy,(X),NLx)]
#ifdef USE_CONST
#define C(I)        _C_const[((I)-1)]
#define D(I)        _D_const[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#else
#define C(I)        _C[((I)-1)]
#define D(I)        _D[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#endif
#define E(Z,Y,X)    _E[CALC_INDEX3((Z),PNLz,(Y),PNLy,(X),PNLx)]
#define F(Z,Y,X,KB) _F[CALC_INDEX4((Z),PNLz,(Y),PNLy,(X),PNLx,(KB),Nkb)]

#define E_IDX(DT)  _E[CALC_INDEX3(iz,PNLz,iy,PNLy,modx[ix+(DT)+NLx],PNLx)]
#define E_IDY(DT)  _E[CALC_INDEX3(iz,PNLz,mody[iy+(DT)+NLy],PNLy,ix,PNLx)]
#define E_IDZ(DT)  _E[CALC_INDEX3(modz[iz+(DT)+NLz],PNLz,iy,PNLy,ix,PNLx)]

__global__ void hpsi1_rt_stencil_kern_v5_1(int Nkb,
					const double * __restrict__ _A, const double * __restrict__ _B, const double * __restrict__ _C, const double * __restrict__ _D,
					const cuDoubleComplex * __E, cuDoubleComplex *_F,
					   int PNLx, int PNLy, int PNLz,
					int NLx, int NLy, int NLz,
					const int * __restrict__ modx, const int * __restrict__ mody, const int * __restrict__ modz )
{
    int ikb, iyz,iz,iy,ix;
    cuDoubleComplex v,w, val;

    ikb = blockIdx.y;
    if ( ikb >= Nkb ) return;

    const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

    iyz = threadIdx.x + blockDim.x * blockIdx.x;
    iz = iyz % NLz;
    iy = iyz / NLz;
    if ( iy >= NLy ) return;

    cuDoubleComplex E_idx[9];

    ix = 0;
    E_idx[0] = E_IDX(-4);
    E_idx[1] = E_IDX(-3);
    E_idx[2] = E_IDX(-2);
    E_idx[3] = E_IDX(-1);
    E_idx[4] = E_IDX( 0);
    E_idx[5] = E_IDX( 1);
    E_idx[6] = E_IDX( 2);
    E_idx[7] = E_IDX( 3);
    E_idx[8] = E_IDX( 4);

    for (ix=0; ix < NLx; ix++) {

	v =         cuCmul(C( 1),(cuCadd(E_idx[5],E_idx[3])));
	v = cuCadd( cuCmul(C( 2),(cuCadd(E_idx[6],E_idx[2]))), v );
	v = cuCadd( cuCmul(C( 3),(cuCadd(E_idx[7],E_idx[1]))), v );
	v = cuCadd( cuCmul(C( 4),(cuCadd(E_idx[8],E_idx[0]))), v );

	w =         cuCmul(D( 1),(cuCsub(E_idx[5],E_idx[3])));
	w = cuCadd( cuCmul(D( 2),(cuCsub(E_idx[6],E_idx[2]))), w );
	w = cuCadd( cuCmul(D( 3),(cuCsub(E_idx[7],E_idx[1]))), w );
	w = cuCadd( cuCmul(D( 4),(cuCsub(E_idx[8],E_idx[0]))), w );

	val = cuCmul((B(iz,iy,ix)+A(ikb)), E_idx[4]);
	val = cuCsub( val, cuCmul(0.5, v) );
	val = cuCsub( val, cuCmulI(w) );

	F(iz,iy,ix,ikb) = val;

	E_idx[0] = E_idx[1];
	E_idx[1] = E_idx[2];
	E_idx[2] = E_idx[3];
	E_idx[3] = E_idx[4];
	E_idx[4] = E_idx[5];
	E_idx[5] = E_idx[6];
	E_idx[6] = E_idx[7];
	E_idx[7] = E_idx[8];
	E_idx[8] = E_IDX(5);
    }
}

#define E_SMEM_IDY(DT) _E_smem[CALC_INDEX2(iz,NLz,mody[iy+(DT)+NLy],NLy)]
#define E_SMEM_IDZ(DT) _E_smem[CALC_INDEX2(modz[iz+(DT)+NLz],NLz,iy,NLy)]

/* */
__launch_bounds__(1024,1)
__global__ void hpsi1_rt_stencil_kern_v5_2(int Nkb,
					const double * __restrict__ _A, const double * __restrict__ _B, const double * __restrict__ _C, const double * __restrict__ _D,
					const cuDoubleComplex * __E, cuDoubleComplex *_F,
					   int PNLx, int PNLy, int PNLz,
					int NLx, int NLy, int NLz,
					const int * __restrict__ modx, const int * __restrict__ mody, const int * __restrict__ modz )
{
    int ikb, iyz,iz,iy,ix;
    cuDoubleComplex v,w, val;

    ikb = blockIdx.y;
    if ( ikb >= Nkb ) return;

    const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

    ix = blockIdx.x;
    if ( ix >= NLx ) return;

    cuDoubleComplex *_E_smem = (cuDoubleComplex*) _dyn_smem;

    for (iyz=threadIdx.x; iyz < NLy*NLz; iyz += blockDim.x) {
	iz = iyz % NLz;
	iy = iyz / NLz;
	_E_smem[iyz] = E(iz,iy,ix);
    }
    __syncthreads();

    for (iyz=threadIdx.x; iyz < NLy*NLz; iyz += blockDim.x) {
	iz = iyz % NLz;
	iy = iyz / NLz;

	const int idz_min =  iy    * NLz;
	const int idz_max = (iy+1) * NLz;
	int idzm = (iz-1) + iy * NLz; if (idzm <  idz_min) idzm += NLz; 
	int idzp = (iz+1) + iy * NLz; if (idzp >= idz_max) idzp -= NLz; 
	v =         cuCmul(C( 9),(cuCadd(_E_smem[idzp],_E_smem[idzm])));
	w =         cuCmul(D( 9),(cuCsub(_E_smem[idzp],_E_smem[idzm])));

	idzm -= 1; if (idzm < idz_min)  idzm += NLz;
	idzp += 1; if (idzp >= idz_max) idzp -= NLz;
	v = cuCadd( cuCmul(C(10),(cuCadd(_E_smem[idzp],_E_smem[idzm]))), v );
	w = cuCadd( cuCmul(D(10),(cuCsub(_E_smem[idzp],_E_smem[idzm]))), w );

	idzm -= 1; if (idzm < idz_min)  idzm += NLz;
	idzp += 1; if (idzp >= idz_max) idzp -= NLz;
	v = cuCadd( cuCmul(C(11),(cuCadd(_E_smem[idzp],_E_smem[idzm]))), v );
	w = cuCadd( cuCmul(D(11),(cuCsub(_E_smem[idzp],_E_smem[idzm]))), w );

	idzm -= 1; if (idzm < idz_min)  idzm += NLz;
	idzp += 1; if (idzp >= idz_max) idzp -= NLz;
	v = cuCadd( cuCmul(C(12),(cuCadd(_E_smem[idzp],_E_smem[idzm]))), v );
	w = cuCadd( cuCmul(D(12),(cuCsub(_E_smem[idzp],_E_smem[idzm]))), w );

	int idym = iz + (iy-1) * NLz; if (idym <  0)       idym += NLy*NLz;
	int idyp = iz + (iy+1) * NLz; if (idyp >= NLy*NLz) idyp -= NLy*NLz;
	v = cuCadd( cuCmul(C( 5),(cuCadd(_E_smem[idyp],_E_smem[idym]))), v );
	w = cuCadd( cuCmul(D( 5),(cuCsub(_E_smem[idyp],_E_smem[idym]))), w );

	idym -= NLz; if (idym < 0)        idym += NLy*NLz;
	idyp += NLz; if (idyp >= NLy*NLz) idyp -= NLy*NLz;
	v = cuCadd( cuCmul(C( 6),(cuCadd(_E_smem[idyp],_E_smem[idym]))), v );
	w = cuCadd( cuCmul(D( 6),(cuCsub(_E_smem[idyp],_E_smem[idym]))), w );

	idym -= NLz; if (idym < 0)        idym += NLy*NLz;
	idyp += NLz; if (idyp >= NLy*NLz) idyp -= NLy*NLz;
	v = cuCadd( cuCmul(C( 7),(cuCadd(_E_smem[idyp],_E_smem[idym]))), v );
	w = cuCadd( cuCmul(D( 7),(cuCsub(_E_smem[idyp],_E_smem[idym]))), w );

	idym -= NLz; if (idym < 0)        idym += NLy*NLz;
	idyp += NLz; if (idyp >= NLy*NLz) idyp -= NLy*NLz;
	v = cuCadd( cuCmul(C( 8),(cuCadd(_E_smem[idyp],_E_smem[idym]))), v );
	w = cuCadd( cuCmul(D( 8),(cuCsub(_E_smem[idyp],_E_smem[idym]))), w );

	val = cuCmul(-0.5, v);
	val = cuCsub( val, cuCmulI(w) );

	F(iz,iy,ix,ikb) = cuCadd(F(iz,iy,ix,ikb), val);
    }
}
#endif

/************************************************************/

#ifdef USE_V8

#define A(KB)       _A[(KB)]
#define B(Z,Y,X)    _B[CALC_INDEX3((Z),NLz,(Y),NLy,(X),NLx)]
#ifdef USE_CONST
#define C(I)        _C_const[((I)-1)]
#define D(I)        _D_const[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#else
#define C(I)        _C[((I)-1)]
#define D(I)        _D[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#endif
#define E(Z,Y,X)    _E[CALC_INDEX3((Z),PNLz,(Y),PNLy,(X),PNLx)]
#define F(Z,Y,X,KB) _F[CALC_INDEX4((Z),PNLz,(Y),PNLy,(X),PNLx,(KB),Nkb)]

#define E_IDX(DT)  _E[CALC_INDEX3(iz,PNLz,iy,PNLy,modx[ix+(DT)+NLx],PNLx)]
#define E_IDY(DT)  _E[CALC_INDEX3(iz,PNLz,mody[iy+(DT)+NLy],PNLy,ix,PNLx)]
#define E_IDZ(DT)  _E[CALC_INDEX3(modz[iz+(DT)+NLz],PNLz,iy,PNLy,ix,PNLx)]

template<int NLX> // __launch_bounds__(128,2)
__global__ void hpsi1_rt_stencil_kern_v8_1(int Nkb,
					const double * __restrict__ _A, const double * __restrict__ _B, const double * __restrict__ _C, const double * __restrict__ _D,
					const cuDoubleComplex * __E, cuDoubleComplex *_F,
					int PNLx, int PNLy, int PNLz,
					int NLx, int NLy, int NLz,
					const int * __restrict__ modx, const int * __restrict__ mody, const int * __restrict__ modz )
{
    int ikb, iyz,iz,iy,ix;

    ikb = blockIdx.y;
    if ( ikb >= Nkb ) return;

    const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

    iyz = threadIdx.x + blockDim.x * blockIdx.x;
    iz = iyz % NLz;
    iy = iyz / NLz;
    if ( iy >= NLy ) return;

    cuDoubleComplex val[NLX];
    for (ix=0; ix < NLX; ix++) {
	val[ix] = make_cuDoubleComplex( 0.0, 0.0 );
    }

    int ixx;
    #pragma unroll
    for (ix=0; ix < NLX; ix++) {
	cuDoubleComplex E_ix = E(iz,iy,ix);

	ixx = ix - 4; if (ixx < 0) ixx += NLX;
	val[ixx] = cuCadd( val[ixx], cuCsub( cuCmul(-0.5*C(4),E_ix), cuCmulI(cuCmul(D(4),E_ix)) ) );
	if ( ix >= 8 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */
	
	ixx = ix - 3; if (ixx < 0) ixx += NLX;
	val[ixx] = cuCadd( val[ixx], cuCsub( cuCmul(-0.5*C(3),E_ix), cuCmulI(cuCmul(D(3),E_ix)) ) );
	if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

	ixx = ix - 2; if (ixx < 0) ixx += NLX;
	val[ixx] = cuCadd( val[ixx], cuCsub( cuCmul(-0.5*C(2),E_ix), cuCmulI(cuCmul(D(2),E_ix)) ) );
	if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

	ixx = ix - 1; if (ixx < 0) ixx += NLX;
	val[ixx] = cuCadd( val[ixx], cuCsub( cuCmul(-0.5*C(1),E_ix), cuCmulI(cuCmul(D(1),E_ix)) ) );
	if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

	ixx = ix;
	val[ixx] = cuCadd( val[ixx], cuCmul((B(iz,iy,ix)+A(ikb)), E_ix) );
	if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

	ixx = ix + 1; if (ixx >= NLX) ixx -= NLX;
	val[ixx] = cuCadd( val[ixx], cuCadd( cuCmul(-0.5*C(1),E_ix), cuCmulI(cuCmul(D(1),E_ix)) ) );
	if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

	ixx = ix + 2; if (ixx >= NLX) ixx -= NLX;
	val[ixx] = cuCadd( val[ixx], cuCadd( cuCmul(-0.5*C(2),E_ix), cuCmulI(cuCmul(D(2),E_ix)) ) );
	if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

	ixx = ix + 3; if (ixx >= NLX) ixx -= NLX;
	val[ixx] = cuCadd( val[ixx], cuCadd( cuCmul(-0.5*C(3),E_ix), cuCmulI(cuCmul(D(3),E_ix)) ) );
	if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */

	ixx = ix + 4; if (ixx >= NLX) ixx -= NLX;
	val[ixx] = cuCadd( val[ixx], cuCadd( cuCmul(-0.5*C(4),E_ix), cuCmulI(cuCmul(D(4),E_ix)) ) );
	if ( ix >= NLX-1 ) F(iz,iy,ixx,ikb) = val[ixx]; /* */
    }
}

__launch_bounds__(128,4)
__global__ void hpsi1_rt_stencil_kern_v8_2(int Nkb,
					const double * __restrict__ _A, const double * __restrict__ _B, const double * __restrict__ _C, const double * __restrict__ _D,
					const cuDoubleComplex * __restrict__ __E, cuDoubleComplex *_F,
					int PNLx, int PNLy, int PNLz,
					int NLx, int NLy, int NLz,
					const int * __restrict__ modx, const int * __restrict__ mody, const int * __restrict__ modz )
{
    int ikb, ixz,iz,iy,ix;
    cuDoubleComplex v,w, val;

    ikb = blockIdx.y;
    if ( ikb >= Nkb ) return;

    const cuDoubleComplex *_E = __E + ikb * (PNLx*PNLy*PNLz);

    ixz = threadIdx.x + blockDim.x * blockIdx.x;
    iz = ixz % NLz;
    ix = ixz / NLz;
    if ( ix >= NLx ) return;

    cuDoubleComplex E_idy[9];

    iy = 0;
    int  idy     = CALC_INDEX3(iz,PNLz,mody[iy+(-4)+NLy],PNLy,ix,PNLx);
    int  idy_min = CALC_INDEX3(0,PNLz,0  ,PNLy,ix,PNLx);
    int  idy_max = CALC_INDEX3(0,PNLz,NLy,PNLy,ix,PNLx);
    E_idy[0] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
    E_idy[1] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
    E_idy[2] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
    E_idy[3] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
    E_idy[4] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
    E_idy[5] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
    E_idy[6] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;
    E_idy[7] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;

    for (iy=0; iy < NLy; iy++) {
	E_idy[8] = _E[idy]; idy += PNLz; if (idy >= idy_max) idy -= NLy*PNLz;

	v =         cuCmul(C(12),(cuCadd(E_IDZ(4),E_IDZ(-4))));
	w =         cuCmul(D(12),(cuCsub(E_IDZ(4),E_IDZ(-4))));
	v = cuCadd( cuCmul(C(11),(cuCadd(E_IDZ(3),E_IDZ(-3)))), v );
	w = cuCadd( cuCmul(D(11),(cuCsub(E_IDZ(3),E_IDZ(-3)))), w );
	v = cuCadd( cuCmul(C(10),(cuCadd(E_IDZ(2),E_IDZ(-2)))), v );
	w = cuCadd( cuCmul(D(10),(cuCsub(E_IDZ(2),E_IDZ(-2)))), w );
	v = cuCadd( cuCmul(C( 9),(cuCadd(E_IDZ(1),E_IDZ(-1)))), v );
	w = cuCadd( cuCmul(D( 9),(cuCsub(E_IDZ(1),E_IDZ(-1)))), w );

	v = cuCadd( cuCmul(C( 5),(cuCadd(E_idy[5],E_idy[3]))), v );
	w = cuCadd( cuCmul(D( 5),(cuCsub(E_idy[5],E_idy[3]))), w );
	v = cuCadd( cuCmul(C( 6),(cuCadd(E_idy[6],E_idy[2]))), v );
	w = cuCadd( cuCmul(D( 6),(cuCsub(E_idy[6],E_idy[2]))), w );
	v = cuCadd( cuCmul(C( 7),(cuCadd(E_idy[7],E_idy[1]))), v );
	w = cuCadd( cuCmul(D( 7),(cuCsub(E_idy[7],E_idy[1]))), w );
	v = cuCadd( cuCmul(C( 8),(cuCadd(E_idy[8],E_idy[0]))), v );
	w = cuCadd( cuCmul(D( 8),(cuCsub(E_idy[8],E_idy[0]))), w );

	val = cuCmul(-0.5, v);
	val = cuCsub( val, cuCmulI(w) );

	F(iz,iy,ix,ikb) = cuCadd(F(iz,iy,ix,ikb), val);

	E_idy[0] = E_idy[1];
	E_idy[1] = E_idy[2];
	E_idy[2] = E_idy[3];
	E_idy[3] = E_idy[4];
	E_idy[4] = E_idy[5];
	E_idy[5] = E_idy[6];
	E_idy[6] = E_idy[7];
	E_idy[7] = E_idy[8];
    }
}
#endif

/************************************************************/

/*
 *
 */
void hpsi1_rt_stencil_gpu(double *_A,  // k2lap0_2(:)
			  double *_B,  // Vloc
			  double *_C,  // lapt(1:12)
			  double *_D,  // nabt(1:12, ikb_s:ikb_e)
			  cuDoubleComplex *_E,  //  tpsi(0:PNL-1, ikb_s:ikb_e)
			  cuDoubleComplex *_F,  // htpsi(0:PNL-1, ikb_s:ikb_e)
			  int IKB_s, int IKB_e, 
			  int PNLx, int PNLy, int PNLz,
			  int NLx, int NLy, int NLz,
			  int *modx, int *mody, int *modz, int myrank )
{
    int Nkb = IKB_e - IKB_s + 1;

#ifdef USE_CONST
    CUDA_CALL( cudaMemcpyToSymbolAsync( _C_const, _C, sizeof(double)*12,     0, cudaMemcpyDeviceToDevice, 0 ) );

    assert( Nkb <= MAX_NKB );
    CUDA_CALL( cudaMemcpyToSymbolAsync( _D_const, _D, sizeof(double)*12*Nkb, 0, cudaMemcpyDeviceToDevice, 0 ) );
#endif

#ifdef USE_V1
    {
	dim3 ts(128,1,1);
	dim3 bs(DIV_CEIL((NLy*NLz),ts.x),Nkb,1);
	hpsi1_rt_stencil_kern_v1<<< bs, ts >>>( Nkb, _A, _B, _C, _D, _E, _F,
						PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );
	CUDA_CALL( cudaDeviceSynchronize() );
    }
#endif

#ifdef USE_V2
    {
	dim3 ts(128,1,1);
	dim3 bs(DIV_CEIL((NLy*NLz),ts.x),Nkb,1);
	hpsi1_rt_stencil_kern_v2<<< bs, ts >>>( Nkb, _A, _B, _C, _D, _E, _F,
						PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );
	CUDA_CALL( cudaDeviceSynchronize() );
    }
#endif

#ifdef USE_V4
    {
	dim3 ts_1(128,1,1);
	dim3 bs_1(DIV_CEIL((NLy*NLz),ts_1.x),Nkb,1);
	hpsi1_rt_stencil_kern_v4_1<<< bs_1, ts_1 >>>( Nkb, _A, _B, _C, _D, _E, _F,
						      PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );

	dim3 ts_2(128,1,1);
	dim3 bs_2(DIV_CEIL((NLx*NLz),ts_2.x),Nkb,1);
	hpsi1_rt_stencil_kern_v4_2<<< bs_2, ts_2 >>>( Nkb, _A, _B, _C, _D, _E, _F,
						      PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );
	CUDA_CALL( cudaDeviceSynchronize() );
    }
#endif

#ifdef USE_V5
    {
	dim3 ts_1(128,1,1);
	dim3 bs_1(DIV_CEIL((NLy*NLz),ts_1.x),Nkb,1);
	hpsi1_rt_stencil_kern_v5_1<<< bs_1, ts_1 >>>( Nkb, _A, _B, _C, _D, _E, _F, 
						      PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );

	dim3 ts_2(1024,1,1);
	while (ts_2.x > NLy*NLz) { ts_2.x /= 2;	}
	dim3 bs_2(NLx,Nkb,1);
	size_t  smem_size = sizeof(cuDoubleComplex)*NLy*NLz;
	assert(smem_size <= 48*1024);
	hpsi1_rt_stencil_kern_v5_2<<< bs_2, ts_2, smem_size >>>( Nkb, _A, _B, _C, _D, _E, _F,
								 PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );
	CUDA_CALL( cudaDeviceSynchronize() );
    }
#endif

#ifdef USE_V8
    {
	dim3 ts_1(128,1,1);
	dim3 bs_1(DIV_CEIL((NLy*NLz),ts_1.x),Nkb,1);
	if ( 0 ) {}
	else if ( NLx == 20 ) {
	    hpsi1_rt_stencil_kern_v8_1<20><<< bs_1, ts_1 >>>( Nkb, _A, _B, _C, _D, _E, _F,
							      PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );
	}
	else if ( NLx == 16 ) {
	    hpsi1_rt_stencil_kern_v8_1<16><<< bs_1, ts_1 >>>( Nkb, _A, _B, _C, _D, _E, _F,
							      PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );
	}
	else { exit( -1 ); }

	dim3 ts_2(128,1,1);
	dim3 bs_2(DIV_CEIL((NLx*NLz),ts_2.x),Nkb,1);
	hpsi1_rt_stencil_kern_v8_2<<< bs_2, ts_2 >>>( Nkb, _A, _B, _C, _D, _E, _F,
						      PNLx, PNLy, PNLz, NLx, NLy, NLz, modx, mody, modz );
	CUDA_CALL( cudaDeviceSynchronize() );
    }
#endif

#ifdef USE_CPU

    /* reference CPU implementation */

#define A(KB)       _A[(KB)]
#define B(Z,Y,X)    _B[CALC_INDEX3((Z),NLz,(Y),NLy,(X),NLx)]
#define C(I)        _C[((I)-1)]
#define D(I)        _D[CALC_INDEX2((I)-1,12,ikb,Nkb)]
#define E(Z,Y,X,KB) _E[CALC_INDEX4((Z),PNLz,(Y),PNLy,(X),PNLx,(KB),Nkb)]
#define F(Z,Y,X,KB) _F[CALC_INDEX4((Z),PNLz,(Y),PNLy,(X),PNLx,(KB),Nkb)]

#define E_IDX(DT)  _E[CALC_INDEX4(iz,PNLz,iy,PNLy,modx[ix+(DT)+NLx],PNLx,ikb,Nkb)]
#define E_IDY(DT)  _E[CALC_INDEX4(iz,PNLz,mody[iy+(DT)+NLy],PNLy,ix,PNLx,ikb,Nkb)]
#define E_IDZ(DT)  _E[CALC_INDEX4(modz[iz+(DT)+NLz],PNLz,iy,PNLy,ix,PNLx,ikb,Nkb)]

    int ikb, iz,iy,ix;
    cuDoubleComplex v,w, val;

    for (ikb=0; ikb < Nkb; ikb++) {
	for (ix=0; ix < NLx; ix++) {
	for (iy=0; iy < NLy; iy++) {
	for (iz=0; iz < NLz; iz++) {

	    v =         cuCmul(C( 9),(cuCadd(E_IDZ(1),E_IDZ(-1))));
	    v = cuCadd( cuCmul(C(10),(cuCadd(E_IDZ(2),E_IDZ(-2)))), v );
	    v = cuCadd( cuCmul(C(11),(cuCadd(E_IDZ(3),E_IDZ(-3)))), v );
	    v = cuCadd( cuCmul(C(12),(cuCadd(E_IDZ(4),E_IDZ(-4)))), v );

	    w =         cuCmul(D( 9),(cuCsub(E_IDZ(1),E_IDZ(-1))));
	    w = cuCadd( cuCmul(D(10),(cuCsub(E_IDZ(2),E_IDZ(-2)))), w );
	    w = cuCadd( cuCmul(D(11),(cuCsub(E_IDZ(3),E_IDZ(-3)))), w );
	    w = cuCadd( cuCmul(D(12),(cuCsub(E_IDZ(4),E_IDZ(-4)))), w );

	    v = cuCadd( cuCmul(C( 5),(cuCadd(E_IDY(1),E_IDY(-1)))), v );
	    v = cuCadd( cuCmul(C( 6),(cuCadd(E_IDY(2),E_IDY(-2)))), v );
	    v = cuCadd( cuCmul(C( 7),(cuCadd(E_IDY(3),E_IDY(-3)))), v );
	    v = cuCadd( cuCmul(C( 8),(cuCadd(E_IDY(4),E_IDY(-4)))), v );

	    w = cuCadd( cuCmul(D( 5),(cuCsub(E_IDY(1),E_IDY(-1)))), w );
	    w = cuCadd( cuCmul(D( 6),(cuCsub(E_IDY(2),E_IDY(-2)))), w );
	    w = cuCadd( cuCmul(D( 7),(cuCsub(E_IDY(3),E_IDY(-3)))), w );
	    w = cuCadd( cuCmul(D( 8),(cuCsub(E_IDY(4),E_IDY(-4)))), w );

	    v = cuCadd( cuCmul(C( 1),(cuCadd(E_IDX(1),E_IDX(-1)))), v );
	    v = cuCadd( cuCmul(C( 2),(cuCadd(E_IDX(2),E_IDX(-2)))), v );
	    v = cuCadd( cuCmul(C( 3),(cuCadd(E_IDX(3),E_IDX(-3)))), v );
	    v = cuCadd( cuCmul(C( 4),(cuCadd(E_IDX(4),E_IDX(-4)))), v );

	    w = cuCadd( cuCmul(D( 1),(cuCsub(E_IDX(1),E_IDX(-1)))), w );
	    w = cuCadd( cuCmul(D( 2),(cuCsub(E_IDX(2),E_IDX(-2)))), w );
	    w = cuCadd( cuCmul(D( 3),(cuCsub(E_IDX(3),E_IDX(-3)))), w );
	    w = cuCadd( cuCmul(D( 4),(cuCsub(E_IDX(4),E_IDX(-4)))), w );

	    val = cuCmul((B(iz,iy,ix)+A(ikb)), E(iz,iy,ix,ikb));
	    val = cuCsub( val, cuCmul(0.5, v) );
	    val = make_cuDoubleComplex( cuCreal(val) + cuCimag(w), cuCimag(val) - cuCreal(w) );

	    F(iz,iy,ix,ikb) = val;
	}}}
    }

    if ( 0 && myrank == 0 ) {
	// for debug
	printf( "IKB_s:%d, IKB_e:%d\n", IKB_s, IKB_e );
	printf( "PNLx:%d, PNLy:%d, PNLz:%d\n", PNLx, PNLy, PNLz );
	printf( "NLx:%d, NLy:%d, NLz:%d\n", NLx, NLy, NLz );

	printf( "modx:" );
	for ( int i = 0; i <= NLx*2 + 4; i++ ) {
	    printf( " %d", modx[i] );
	}
	printf( "\n" );

	printf( "mody:" );
	for ( int i = 0; i <= NLy*2 + 4; i++ ) {
	    printf( " %d", mody[i] );
	}
	printf( "\n" );

	printf( "modz:" );
	for ( int i = 0; i <= NLz*2 + 4; i++ ) {
	    printf( " %d", modz[i] );
	}
	printf( "\n" );

	printf( "A:" );
	for ( int i = 0; i < Nkb ; i++ ) {
	    printf( " %lf", A(i) );
	}
	printf( "\n" );

	for ( int i = 0; i < NLx; i++ ) {
	    ix = iy = iz = i;
	    double val = B(iz,iy,ix);
	    printf( "B(%d,%d,%d): %lf\n", iz, iy, ix, val );
	}

	printf( "C:" );
	for ( int i = 1; i <= 12; i++ ) {
	    printf( " %lf", C(i) );
	}
	printf( "\n" );

	ikb = 0;
	printf( "D:" );
	for ( int i = 1; i <= 12; i++ ) {
	    printf( " %lf", D(i) );
	}
	printf( "\n" );

	for ( int i = 0; i < 100; i++ ) {
	    ikb = i % Nkb; 
	    ix = i % NLx;
	    iy = i % NLy;
	    iz = i % NLz;
	    v = E(iz,iy,ix,ikb);
	    printf( "E(%d,%d,%d,%d): %lf, %lf\n", iz, iy, ix, ikb, v.x, v.y );
	}

	for ( int i = 0; i < 100; i++ ) {
	    ikb = i % Nkb; 
	    ix = i % NLx;
	    iy = i % NLy;
	    iz = i % NLz;
	    v = F(iz,iy,ix,ikb);
	    printf( "F(%d,%d,%d,%d): %lf, %lf\n", iz, iy, ix, ikb, v.x, v.y );
	}
    }
#endif
}

extern "C" {
    void hpsi1_rt_stencil_gpu_(double *A, double *B, double *C, double *D, cuDoubleComplex *E, cuDoubleComplex *F,
			       int *IKB_s, int *IKB_e, 
			       int *PNLx, int *PNLy, int *PNLz,
			       int *NLx, int *NLy, int *NLz,
			       int *modx, int *mody, int *modz, int *myrank ) {
	hpsi1_rt_stencil_gpu(A, B, C, D, E, F,
			     *IKB_s, *IKB_e, 
			     *PNLx, *PNLy, *PNLz,
			     *NLx, *NLy, *NLz,
			     modx, mody, modz, *myrank );
    }
}

