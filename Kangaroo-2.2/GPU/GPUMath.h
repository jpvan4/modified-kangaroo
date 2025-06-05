/*
* This file is part of the BTCCollider distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

// ---------------------------------------------------------------------------------
// 256(+64) bits integer CUDA libray for SECPK1
// ---------------------------------------------------------------------------------



#ifndef GPUMATHH
#define GPUMATHH

#include <cuda.h>
#include <stdint.h>

#define NBBLOCK 5
#define BIFULLSIZE 40

// --- CUDA asm helpers ---
#define UADDO(c,a,b) asm volatile("add.cc.u64 %0,%1,%2;" : "=l"(c) : "l"(a), "l"(b));
#define UADDC(c,a,b) asm volatile("addc.cc.u64 %0,%1,%2;" : "=l"(c) : "l"(a), "l"(b));
#define UADD(c,a,b)  asm volatile("addc.u64 %0,%1,%2;" : "=l"(c) : "l"(a), "l"(b));

#define USUBO(c,a,b) asm volatile("sub.cc.u64 %0,%1,%2;" : "=l"(c) : "l"(a), "l"(b));
#define USUBC(c,a,b) asm volatile("subc.cc.u64 %0,%1,%2;" : "=l"(c) : "l"(a), "l"(b));
#define USUB(c,a,b)  asm volatile("subc.u64 %0,%1,%2;" : "=l"(c) : "l"(a), "l"(b));

#define UMULLO(lo,a,b) asm volatile("mul.lo.u64 %0,%1,%2;" : "=l"(lo) : "l"(a), "l"(b));
#define UMULHI(hi,a,b) asm volatile("mul.hi.u64 %0,%1,%2;" : "=l"(hi) : "l"(a), "l"(b));

#define MADDO(r,a,b,c)  asm volatile("mad.lo.cc.u64 %0,%1,%2,%3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
#define MADDC(r,a,b,c)  asm volatile("madc.lo.cc.u64 %0,%1,%2,%3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
#define MADD(r,a,b,c)   asm volatile("madc.lo.u64 %0,%1,%2,%3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));

#define MM64 0xD838091DD2253531ULL

///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void Add256(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t c;
    UADDO(r[0], a[0], b[0]);
    UADDC(r[1], a[1], b[1]);
    UADDC(r[2], a[2], b[2]);
    UADDC(r[3], a[3], b[3]);
}

__device__ __forceinline__ void Sub256(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t c;
    USUBO(r[0], a[0], b[0]);
    USUBC(r[1], a[1], b[1]);
    USUBC(r[2], a[2], b[2]);
    USUBC(r[3], a[3], b[3]);
}

///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void _ModMult(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t p[9]={0}, c1,c2;

#define MULACC(i,j,idx)                        \
    { uint64_t lo, hi;                         \
      UMULLO(lo, a[i], b[j]);                  \
      UMULHI(hi, a[i], b[j]);                  \
      uint64_t k=idx;                          \
      UADDO(c1, p[k], lo); p[k]=c1;            \
      UADDC(c2, p[k+1], hi);                   \
      UADD(p[k+2], p[k+2], c2);                \
    }

    MULACC(0,0,0);

    MULACC(0,1,1);
    MULACC(1,0,1);

    MULACC(0,2,2);
    MULACC(1,1,2);
    MULACC(2,0,2);

    MULACC(0,3,3);
    MULACC(1,2,3);
    MULACC(2,1,3);
    MULACC(3,0,3);

    MULACC(1,3,4);
    MULACC(2,2,4);
    MULACC(3,1,4);

    MULACC(2,3,5);
    MULACC(3,2,5);

    MULACC(3,3,6);

#undef MULACC

    for(int i=4;i<=8;i++){
        uint64_t t[5];
        UMult(t,(p+i),0x1000003D1ULL);
        UADDO(p[i-4],p[i-4],t[0])
        UADDC(p[i-3],p[i-3],t[1])
        UADDC(p[i-2],p[i-2],t[2])
        UADDC(p[i-1],p[i-1],t[3])
    }

    uint64_t mp,mh;
    UMULLO(mp,p[4],0x1000003D1ULL);
    UMULHI(mh,p[4],0x1000003D1ULL);
    UADDO(r[0],p[0],mp);
    UADDC(r[1],p[1],mh);
    UADDC(r[2],p[2],0);
    UADD(r[3],p[3],0);
}

///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void _ModSqr(uint64_t *r, const uint64_t *a) {
    _ModMult(r, a, a);
}

///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void Add128(uint64_t *r, const uint64_t *b){
    uint64_t c;
    UADDO(r[0],r[0],b[0]);
    UADD(r[1],r[1],b[1]);
}

///////////////////////////////////////////////////////////////////////////////
// fast sub+conditional add P (mod reduction)
__device__ __forceinline__ void ModSub256(uint64_t* r, const uint64_t* a, const uint64_t* b){
    uint64_t t,c;
    USUBO(r[0],a[0],b[0]);
    USUBC(r[1],a[1],b[1]);
    USUBC(r[2],a[2],b[2]);
    USUBC(r[3],a[3],b[3]);
    USUB(t,0,0);
    t = ~t +1; // mask = (borrow) ? 0xFFFFFFFFFFFFFFFF : 0
    uint64_t P[4]={0xFFFFFFFEFFFFFC2FULL,0xFFFFFFFFFFFFFFFFULL,
                      0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL};
    UADDO1(r[0], P[0] & t);
    UADDC1(r[1], P[1] & t);
    UADDC1(r[2], P[2] & t);
    UADD1(r[3], P[3] & t);
}
/////////////////////////

__device__ __forceinline__ void ModSub256(uint64_t* r,const uint64_t* b)
{
    uint64_t t, c;
    USUBO(r[0],r[0],b[0]);
    USUBC(r[1],r[1],b[1]);
    USUBC(r[2],r[2],b[2]);
    USUBC(r[3],r[3],b[3]);
    USUB(t,0,0);
    t=~t+1;
    uint64_t P[4]={0xFFFFFFFEFFFFFC2FULL,0xFFFFFFFFFFFFFFFFULL,
                      0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL};
    UADDO1(r[0],P[0]&t);
    UADDC1(r[1],P[1]&t);
    UADDC1(r[2],P[2]&t);
    UADD1(r[3],P[3]&t);
}

///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void _ModMult(uint64_t *r, uint64_t *a)
{
    _ModMult(r, r, a);
}

///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void _ModInvGrouped(uint64_t r[GPU_GRP_SIZE][4]){
    uint64_t acc[4],tmp[4],inv[5],newV[4];

    Load256(acc,r[0]);
    for(uint32_t i=1;i<GPU_GRP_SIZE;++i)
        _ModMult(acc,acc,r[i]);

    inv[4]=0;
    Load256(inv, acc);
    _ModInv(inv);

    for(int i=GPU_GRP_SIZE-1;i>0;i--){
        _ModMult(newV,inv,r[i-1]);
        _ModMult(inv,inv,r[i]);
        Load256(r[i],newV);
    }
    Load256(r[0],inv);
}

///////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void _ModInv(uint64_t *R){

  // Compute modular inverse of R mop P (using 320bits signed integer)
  // 0 < this < P  , P must be odd
  // Return 0 if no inverse

  int64_t  uu,uv,vu,vv;
  uint64_t r0,s0;
  int64_t  eta = -1;

  uint64_t u[NBBLOCK];
  uint64_t v[NBBLOCK];
  uint64_t r[NBBLOCK];
  uint64_t s[NBBLOCK];
  uint64_t t1[NBBLOCK];
  uint64_t t2[NBBLOCK];
  uint64_t t3[NBBLOCK];
  uint64_t t4[NBBLOCK];

  u[0] = 0xFFFFFFFEFFFFFC2F;
  u[1] = 0xFFFFFFFFFFFFFFFF;
  u[2] = 0xFFFFFFFFFFFFFFFF;
  u[3] = 0xFFFFFFFFFFFFFFFF;
  u[4] = 0;
  Load(v,R);

  // Delayed right shift 62bits
  // Do not maintain a matrix for r and s, the number of 
  // 'added P' can be easily calculated

  // Fist step (r,s)=(0,1) ----------------------------

  uu = 1; uv = 0;
  vu = 0; vv = 1;

  _DivStep62((int64_t)u[0],(int64_t)v[0],&eta,&uu,&uv,&vu,&vv);

  // Now update BigInt variables

  // u = (uu*u + uv*v)
  // v = (vu*u + vv*v)
  IMult(t1,u,uu);
  IMult(t2,v,uv);
  IMult(t3,u,vu);
  IMult(t4,v,vv);
  Add2(u,t1,t2);
  Add2(v,t3,t4);

  _LoadI64(t2,uv);
  _LoadI64(t4,vv);

  // Compute multiple of P to add to s and r to make them multiple of 2^62
  r0 = (t2[0] * MM64) & MSK62;
  s0 = (t4[0] * MM64) & MSK62;
  MulP(r,r0);
  Add1(r,t2);
  MulP(s,s0);
  Add1(s,t4);

  // Right shift all variables by 62bits
  ShiftR62(u);
  ShiftR62(v);
  ShiftR62(r);
  ShiftR62(s);

  // DivStep loop -------------------------------

  while(true) {

    uu = 1; uv = 0;
    vu = 0; vv = 1;

    _DivStep62((int64_t)u[0],(int64_t)v[0],&eta,&uu,&uv,&vu,&vv);

    // Now update BigInt variables

    // u = (uu*u + uv*v)
    // v = (vu*u + vv*v)
    IMult(t1,u,uu);
    IMult(t2,v,uv);
    IMult(t3,u,vu);
    IMult(t4,v,vv);
    Add2(u,t1,t2);
    Add2(v,t3,t4);

    // Right shift (u,v) by 62bits
    ShiftR62(u);
    ShiftR62(v);

    IMult(t1,r,uu);
    IMult(t2,s,uv);

    if(_IsZero(v)) {

      // Last step
      // s not needed
      r0 = ((t1[0] + t2[0]) * MM64) & MSK62;
      MulP(r,r0);
      Add1(r,t1);
      Add1(r,t2);
      ShiftR62(r);
      break;

    } else {

      // r = (uu*r + uv*s + r0*P)
      // s = (vu*r + vv*s + s0*P)

      IMult(t3,r,vu);
      IMult(t4,s,vv);

      // Compute multiple of P to add to s to make it multiple of 2^62
      r0 = ((t1[0] + t2[0]) * MM64) & MSK62;
      s0 = ((t3[0] + t4[0]) * MM64) & MSK62;
      MulP(r,r0);
      Add1(r,t1);
      Add1(r,t2);

      // s = (vu*r + vv*s + s0*P)
      MulP(s,s0);
      Add1(s,t3);
      Add1(s,t4);

      // Right shift (r,s) by 62bits
      ShiftR62(r);
      ShiftR62(s);

    }

  }

  // u ends with -1 or 1
  if(_IsNegative(u)) {
    Neg(u);
    Neg(r);
  }

  if(!_IsOne(u)) {
    // No inverse
    R[0] = 0ULL;
    R[1] = 0ULL;
    R[2] = 0ULL;
    R[3] = 0ULL;
    R[4] = 0ULL;
    return;
  }

  while(_IsNegative(r))
    AddP(r);
  while(!_IsNegative(r))
    SubP(r);
  AddP(r);

  Load(R,r);

  /*
  int64_t msk = (int64_t)(u[4]) >> 63;
  int64_t nmsk = ~msk;
  USUBO(r[0],r[0] & nmsk,r[0] & msk);
  USUBC(r[1],r[1] & nmsk,r[1] & msk);
  USUBC(r[2],r[2] & nmsk,r[2] & msk);
  USUBC(r[3],r[3] & nmsk,r[3] & msk);
  USUB(r[4],r[4] & nmsk,r[4] & msk);

  Add16P(r);
  // Reduce from 320 to 256 
  uint64_t ah;
  uint64_t al;
  UMULLO(al,r[4],0x1000003D1ULL);
  UMULHI(ah,r[4],0x1000003D1ULL);
  UADDO(R[0],r[0],al);
  UADDC(R[1],r[1],ah);
  UADDC(R[2],r[2],0ULL);
  UADD(R[3],r[3],0ULL);
  */

}

// ---------------------------------------------------------------------------------------
// Compute a*b*(mod n)
// a and b must be lower than n
// ---------------------------------------------------------------------------------------

__device__ void _ModMult(uint64_t *r,uint64_t *a,uint64_t *b) {

  uint64_t r512[8];
  uint64_t t[NBBLOCK];
  uint64_t ah,al;

  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;

  // 256*256 multiplier
  UMult(r512,a,b[0]);
  UMult(t,a,b[1]);
  UADDO1(r512[1],t[0]);
  UADDC1(r512[2],t[1]);
  UADDC1(r512[3],t[2]);
  UADDC1(r512[4],t[3]);
  UADD1(r512[5],t[4]);
  UMult(t,a,b[2]);
  UADDO1(r512[2],t[0]);
  UADDC1(r512[3],t[1]);
  UADDC1(r512[4],t[2]);
  UADDC1(r512[5],t[3]);
  UADD1(r512[6],t[4]);
  UMult(t,a,b[3]);
  UADDO1(r512[3],t[0]);
  UADDC1(r512[4],t[1]);
  UADDC1(r512[5],t[2]);
  UADDC1(r512[6],t[3]);
  UADD1(r512[7],t[4]);
  // Reduce from 512 to 320 
  UMult(t,(r512 + 4),0x1000003D1ULL);
  UADDO1(r512[0],t[0]);
  UADDC1(r512[1],t[1]);
  UADDC1(r512[2],t[2]);
  UADDC1(r512[3],t[3]);

  // Reduce from 320 to 256 
  UADD1(t[4],0ULL);
  UMULLO(al,t[4],0x1000003D1ULL);
  UMULHI(ah,t[4],0x1000003D1ULL);
  UADDO(r[0],r512[0],al);
  UADDC(r[1],r512[1],ah);
  UADDC(r[2],r512[2],0ULL);
  UADD(r[3],r512[3],0ULL);

}


__device__ void _ModMult(uint64_t *r,uint64_t *a) {

  uint64_t r512[8];
  uint64_t t[NBBLOCK];
  uint64_t ah,al;
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;

  // 256*256 multiplier
  UMult(r512,a,r[0]);
  UMult(t,a,r[1]);
  UADDO1(r512[1],t[0]);
  UADDC1(r512[2],t[1]);
  UADDC1(r512[3],t[2]);
  UADDC1(r512[4],t[3]);
  UADD1(r512[5],t[4]);
  UMult(t,a,r[2]);
  UADDO1(r512[2],t[0]);
  UADDC1(r512[3],t[1]);
  UADDC1(r512[4],t[2]);
  UADDC1(r512[5],t[3]);
  UADD1(r512[6],t[4]);
  UMult(t,a,r[3]);
  UADDO1(r512[3],t[0]);
  UADDC1(r512[4],t[1]);
  UADDC1(r512[5],t[2]);
  UADDC1(r512[6],t[3]);
  UADD1(r512[7],t[4]);

  // Reduce from 512 to 320 
  UMult(t,(r512 + 4),0x1000003D1ULL);
  UADDO1(r512[0],t[0]);
  UADDC1(r512[1],t[1]);
  UADDC1(r512[2],t[2]);
  UADDC1(r512[3],t[3]);

  // Reduce from 320 to 256
  UADD1(t[4],0ULL);
  UMULLO(al,t[4],0x1000003D1ULL);
  UMULHI(ah,t[4],0x1000003D1ULL);
  UADDO(r[0],r512[0],al);
  UADDC(r[1],r512[1],ah);
  UADDC(r[2],r512[2],0ULL);
  UADD(r[3],r512[3],0ULL);

}

__device__ void _ModSqr(uint64_t *rp,const uint64_t *up) {

  uint64_t r512[8];

  uint64_t u10,u11;

  uint64_t r0;
  uint64_t r1;
  uint64_t r3;
  uint64_t r4;

  uint64_t t1;
  uint64_t t2;


  //k=0
  UMULLO(r512[0],up[0],up[0]);
  UMULHI(r1,up[0],up[0]);

  //k=1
  UMULLO(r3,up[0],up[1]);
  UMULHI(r4,up[0],up[1]);
  UADDO1(r3,r3);
  UADDC1(r4,r4);
  UADD(t1,0x0ULL,0x0ULL);
  UADDO1(r3,r1);
  UADDC1(r4,0x0ULL);
  UADD1(t1,0x0ULL);
  r512[1] = r3;

  //k=2
  UMULLO(r0,up[0],up[2]);
  UMULHI(r1,up[0],up[2]);
  UADDO1(r0,r0);
  UADDC1(r1,r1);
  UADD(t2,0x0ULL,0x0ULL);
  UMULLO(u10,up[1],up[1]);
  UMULHI(u11,up[1],up[1]);
  UADDO1(r0,u10);
  UADDC1(r1,u11);
  UADD1(t2,0x0ULL);
  UADDO1(r0,r4);
  UADDC1(r1,t1);
  UADD1(t2,0x0ULL);

  r512[2] = r0;

  //k=3
  UMULLO(r3,up[0],up[3]);
  UMULHI(r4,up[0],up[3]);
  UMULLO(u10,up[1],up[2]);
  UMULHI(u11,up[1],up[2]);
  UADDO1(r3,u10);
  UADDC1(r4,u11);
  UADD(t1,0x0ULL,0x0ULL);
  t1 += t1;
  UADDO1(r3,r3);
  UADDC1(r4,r4);
  UADD1(t1,0x0ULL);
  UADDO1(r3,r1);
  UADDC1(r4,t2);
  UADD1(t1,0x0ULL);

  r512[3] = r3;

  //k=4
  UMULLO(r0,up[1],up[3]);
  UMULHI(r1,up[1],up[3]);
  UADDO1(r0,r0);
  UADDC1(r1,r1);
  UADD(t2,0x0ULL,0x0ULL);
  UMULLO(u10,up[2],up[2]);
  UMULHI(u11,up[2],up[2]);
  UADDO1(r0,u10);
  UADDC1(r1,u11);
  UADD1(t2,0x0ULL);
  UADDO1(r0,r4);
  UADDC1(r1,t1);
  UADD1(t2,0x0ULL);

  r512[4] = r0;

  //k=5
  UMULLO(r3,up[2],up[3]);
  UMULHI(r4,up[2],up[3]);
  UADDO1(r3,r3);
  UADDC1(r4,r4);
  UADD(t1,0x0ULL,0x0ULL);
  UADDO1(r3,r1);
  UADDC1(r4,t2);
  UADD1(t1,0x0ULL);

  r512[5] = r3;

  //k=6
  UMULLO(r0,up[3],up[3]);
  UMULHI(r1,up[3],up[3]);
  UADDO1(r0,r4);
  UADD1(r1,t1);
  r512[6] = r0;

  //k=7
  r512[7] = r1;

#if 1

  // Reduce from 512 to 320 
  UMULLO(r0,r512[4],0x1000003D1ULL);
  UMULLO(r1,r512[5],0x1000003D1ULL);
  MADDO(r1,r512[4],0x1000003D1ULL,r1);
  UMULLO(t2,r512[6],0x1000003D1ULL);
  MADDC(t2,r512[5],0x1000003D1ULL,t2);
  UMULLO(r3,r512[7],0x1000003D1ULL);
  MADDC(r3,r512[6],0x1000003D1ULL,r3);
  MADD(r4,r512[7],0x1000003D1ULL,0ULL);

  UADDO1(r512[0],r0);
  UADDC1(r512[1],r1);
  UADDC1(r512[2],t2);
  UADDC1(r512[3],r3);

  // Reduce from 320 to 256
  UADD1(r4,0ULL);
  UMULLO(u10,r4,0x1000003D1ULL);
  UMULHI(u11,r4,0x1000003D1ULL);
  UADDO(rp[0],r512[0],u10);
  UADDC(rp[1],r512[1],u11);
  UADDC(rp[2],r512[2],0ULL);
  UADD(rp[3],r512[3],0ULL);

#else

  uint64_t z1,z2,z3,z4,z5,z6,z7,z8;

  UMULLO(z3,r512[5],0x1000003d1ULL);
  UMULHI(z4,r512[5],0x1000003d1ULL);
  UMULLO(z5,r512[6],0x1000003d1ULL);
  UMULHI(z6,r512[6],0x1000003d1ULL);
  UMULLO(z7,r512[7],0x1000003d1ULL);
  UMULHI(z8,r512[7],0x1000003d1ULL);
  UMULLO(z1,r512[4],0x1000003d1ULL);
  UMULHI(z2,r512[4],0x1000003d1ULL);
  UADDO1(z1,r512[0]);
  UADD1(z2,0x0ULL);


  UADDO1(z2,r512[1]);
  UADDC1(z4,r512[2]);
  UADDC1(z6,r512[3]);
  UADD1(z8,0x0ULL);

  UADDO1(z3,z2);
  UADDC1(z5,z4);
  UADDC1(z7,z6);
  UADD1(z8,0x0ULL);

  UMULLO(u10,z8,0x1000003d1ULL);
  UMULHI(u11,z8,0x1000003d1ULL);
  UADDO1(z1,u10);
  UADDC1(z3,u11);
  UADDC1(z5,0x0ULL);
  UADD1(z7,0x0ULL);

  rp[0] = z1;
  rp[1] = z3;
  rp[2] = z5;
  rp[3] = z7;

#endif

}

#endif // GPUMATHH

