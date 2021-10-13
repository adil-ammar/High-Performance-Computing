#include "mygemm.h"

/**
 * 
 * Implement all functions here in this file.
 * Do NOT change input parameters and return type.
 * 
 **/

//Register Reuse part 1
//
void dgemm0(const double* A, const double* B, double* C, const int n)
{
int i, j, k;
for (i=0; i<n; i++){
	for (j=0; j<n; j++){
		for (k=0; k<n; k++){
			C[i*n+j] += A[i*n+k] * B[k*n+j];
		}
	}
}
}

void dgemm1(const double *A, const double *B, double *C, const int n) 
{
int i, j, k;
for (i=0; i<n; i++){
	for (j=0; j<n; j++) {
		register double r = C[i*n+j] ;
		for (k=0; k<n; k++)
			r += A[i*n+k] * B[k*n+j];
		C[i*n+j] = r;
	}
}
}
//Register Reuse part 1 End

//Register Reuse part 2
void dgemm2(const double *A, const double *B, double *C, const int n) 
{
int i, j, k;
for(i = 0; i < n; i += 2)
       for(j = 0; j < n; j += 2)  {
            register int t = i*n+j; register int tt = t+n; 
            register double c00 = C[t]; register double c01 = C[t+1];  register double c10 = C[tt]; register double c11 = C[tt+1];

            for(k = 0; k < n; k += 2) {
                // 2 by 2 mini matrix multiplication using registers
                register int ta = i*n+k; register int tta = ta+n; register int tb = k*n+j; register int ttb = tb+n;
                register double a00 = A[ta]; register double a01 = A[ta+1]; register double a10 = A[tta]; register double a11 = A[tta+1];
                register double b00 = B[tb]; register double b01 = B[tb+1]; register double b10 = B[ttb]; register double b11 = B[ttb+1];
                c00 += a00*b00 + a01*b10;
                c01 += a00*b01 + a01*b11;
                c10 += a10*b00 + a11*b10;
                c11 += a10*b01 + a11*b11;
             }

             C[t] = c00;
             C[t+1] = c01;
             C[tt] = c10;
             C[tt+1] = c11;
        }
}
//Register Reuse part 2 End

//Register Reuse part 3
void dgemm3(const double *A, const double *B, double *C, const int n) 
{
//using 15 double registers in total but n needs to be a multiple of 3.
int i, j, k;
for(i = 0; i < n; i+=3){
	for(j = 0; j < n; j+=3){
		//using 9 double registers for 3x3 block of C
		register int tc = i*n+j; register int ttc = tc+n; register int tttc = ttc+n;
		register double c00 = C[tc]; register double c01 = C[tc+1]; register double c02 = C[tc+2];
		register double c10 = C[ttc]; register double c11 = C[ttc+1]; register double c12 = C[ttc+2];
		register double c20 = C[tttc]; register double c21 = C[tttc+1]; register double c22 = C[tttc+2];
		for(k = 0; k < n; k+=3){
			//using 3 double registers to hold three column elements of first row for A and three row elements of first column for B
			register int ta = i*n+k; register int tta = ta+n; register int ttta = tta+n;
			register int tb = k*n+j; register int ttb = tb+n; register int tttb = ttb+n;
			register double a00 = A[ta]; register double a10 = A[tta]; register double a20 = A[ttta];
			register double b00 = B[tb]; register double b01 = B[tb+1]; register double b02 = B[tb+2];
			
			c00 += a00*b00; c01 += a00*b01; c02 += a00*b02;
			c10 += a10*b00; c11 += a10*b01; c12 += a10*b02;
			c20 += a20*b00; c21 += a20*b01; c22 += a20*b02;

			//Similarly holding second column and row in A&B respectively
			a00 = A[ta+1]; a10 = A[tta+1]; a20 = A[ttta+1];
			b00 = B[ttb]; b01 = B[ttb+1]; b02 = B[ttb+2];

                        c00 += a00*b00; c01 += a00*b01; c02 += a00*b02;
                        c10 += a10*b00; c11 += a10*b01; c12 += a10*b02;
                        c20 += a20*b00; c21 += a20*b01; c22 += a20*b02;

			//Similarly holding third column and row in A&B respectively
                        a00 = A[ta+2]; a10 = A[tta+2]; a20 = A[ttta+2];
                        b00 = B[tttb]; b01 = B[tttb+1]; b02 = B[tttb+2];

                        c00 += a00*b00; c01 += a00*b01; c02 += a00*b02;
                        c10 += a10*b00; c11 += a10*b01; c12 += a10*b02;
                        c20 += a20*b00; c21 += a20*b01; c22 += a20*b02;
		}
		C[tc] = c00; C[tc+1] = c01; C[tc+2] = c02;
		C[ttc] = c10; C[ttc+1] = c11; C[ttc+2] = c12;
		C[tttc] = c20; C[tttc+1] = c21; C[tttc+2] = c22;
	}
}
}
//Register Reuse part 3 End

//Cache Reuse part 3
void ijk(const double *A, const double *B, double *C, const int n) 
{
int i, j, k;
for (i=0; i<n; i++){
	for (j=0; j<n; j++){
		register double r=C[i*n+j];
		for (k=0; k<n; k++){
			r += A[i*n+k] * B[k*n+j];
		}
		C[i*n+j]=r;
	}
}
}

void bijk(const double *A, const double *B, double *C, const int n, const int b) 
{
int i, i1, j, j1, k, k1;
for (i = 0; i < n; i+=b){
	for (j = 0; j < n; j+=b){
		for (k = 0; k < n; k+=b){
			// B x B mini matrix multiplications
			for (i1 = i; i1 < i+b; i1++){
				for (j1 = j; j1 < j+b; j1++){
					register double r=C[i1*n+j1];
					for (k1 = k; k1 < k+b; k1++){
						r += A[i1*n + k1]*B[k1*n + j1];
					}
					C[i1*n+j1]=r;
				}
			}
		}
	}
}
}

void jik(const double *A, const double *B, double *C, const int n) 
{
int i, j, k;
for (j=0; j<n; j++){
        for (i=0; i<n; i++){
                register double r=C[i*n+j];
                for (k=0; k<n; k++)
                        r += A[i*n+k] * B[k*n+j];
                C[i*n+j]=r;
        }
}
}

void bjik(const double *A, const double *B, double *C, const int n, const int b) 
{
int i, i1, j, j1, k, k1;
for (j = 0; j < n; j+=b){
        for (i = 0; i < n; i+=b){
                for (k = 0; k < n; k+=b){
                        // B x B mini matrix multiplications
                        for (j1 = j; j1 < j+b && j1 < n; j1++){
                                for (i1 = i; i1 < i+b && i1 < n; i1++){
                                        register double r=C[i1*n+j1];
                                        for (k1 = k; k1 < k+b && k1 < n; k1++){
                                                r += A[i1*n + k1]*B[k1*n + j1];
					}
                                        C[i1*n+j1]=r;
                                }
                        }
                }
        }
}
}

void kij(const double *A, const double *B, double *C, const int n) 
{
int i, j, k;
for (k=0; k<n; k++){
        for (i=0; i<n; i++){
                register double r=A[i*n+k];
                for (j=0; j<n; j++)
                	C[i*n+j] += r * B[k*n+j];
        }
}
}

void bkij(const double *A, const double *B, double *C, const int n, const int b) 
{
int i, i1, j, j1, k, k1;
for (k = 0; k < n; k+=b){
        for (i = 0; i < n; i+=b){
                for (j = 0; j < n; j+=b){
                        // B x B mini matrix multiplications
                        for (k1 = k; k1 < k+b && k1 < n; k1++){
                                for (i1 = i; i1 < i+b && i1 < n; i1++){
                                        register double r=A[i1*n+k1];
                                        for (j1 = j; j1 < j+b && j1 < n; j1++)
                                                C[i1*n+j1] += r*B[k1*n + j1];
                                }
                        }
                }
        }
}
}


void ikj(const double *A, const double *B, double *C, const int n) 
{
int i, j, k;
for (i=0; i<n; i++){
        for (k=0; k<n; k++){
                register double r=A[i*n+k];
                for (j=0; j<n; j++)
                        C[i*n+j] += r * B[k*n+j];
        }
}
}

void bikj(const double *A, const double *B, double *C, const int n, const int b) 
{
int i, i1, j, j1, k, k1;
for (i = 0; i < n; i+=b){
        for (k = 0; k < n; k+=b){
                for (j = 0; j < n; j+=b){
                        // B x B mini matrix multiplications
                        for (i1 = i; i1 < i+b && i1 < n; i1++){
                                for (k1 = k; k1 < k+b && k1 < n; k1++){
                                        register double r=A[i1*n+k1];
                                        for (j1 = j; j1 < j+b && j1 < n; j1++)
                                                C[i1*n+j1] += r*B[k1*n + j1];
                                }
                        }
                }
        }
}
}

void jki(const double *A, const double *B, double *C, const int n) 
{
int i, j, k;
for (j=0; j<n; j++){
        for (k=0; k<n; k++){
                register double r=B[k*n+j];
                for (i=0; i<n; i++)
                        C[i*n+j] += A[i*n+k] * r;
        }
}
}

void bjki(const double *A, const double *B, double *C, const int n, const int b) 
{
int i, i1, j, j1, k, k1;
for (j = 0; j < n; j+=b){
        for (k = 0; k < n; k+=b){
                for (i = 0; i < n; i+=b){
                        //B x B mini matrix multiplications
                        for (j1 = j; j1 < j+b && j1 < n; j1++){
                                for (k1 = k; k1 < k+b && k1 < n; k1++){
                                        register double r=B[k1*n+j1];
                                        for (i1 = i; i1 < i+b && i1 < n; i1++)
                                                C[i1*n+j1] += A[i1*n + k1]*r;
                                }
                        }
                }
        }
}
}

void kji(const double *A, const double *B, double *C, const int n) 
{
/*
int i, j, k;
for (k=0; k<n; k++){
        for (j=0; j<n; j++){
                register double r=B[k*n+j];
                for (i=0; i<n; i++)
                        C[i*n+j] += A[i*n+k] * r;
        }
}
*/
}

void bkji(const double *A, const double *B, double *C, const int n, const int b) 
{
int i, i1, j, j1, k, k1;
for (k = 0; k < n; k+=b){
        for (j = 0; j < n; j+=b){
                for (i = 0; i < n; i+=b){
                        // B x B mini matrix multiplications
                        for (k1 = k; k1 < k+b && k1 < n; k1++){
                                for (j1 = j; j1 < j+b && j1 < n; j1++){
                                        register double r=B[k1*n+j1];
                                        for (i1 = i; i1 < i+b && i1 < n; i1++)
                                                C[i1*n+j1] += A[i1*n + k1]*r;
                                }
                        }
                }
        }
}
}
//Cache Reuse part 3 End 

//Cache Reuse part 4
void optimal(const double* A, const double* B, double *C, const int n, const int b)
{

int i, j, k, i1, j1, k1;
for(i = 0; i < n; i+=b){
        for(j = 0; j < n; j+=b){
                for(k = 0; k < n; k+=b){
                        for(i1 = i; i1 < ((n < i+b) ? n : i+b); i1+=3){
                                for(j1 = j; j1 < ((n < j+b) ? n : j+b); j1+=3){

                                        //using 9 double registers for 3x3 block of C
                                        register double c00 = C[i1 * n + j1]; register double c01 = C[i1 * n + (j1 + 1)]; register double c02 = C[i1 * n + (j1 + 2)];
                                        register double c10 = C[(i1 + 1) * n + j1]; register double c11 = C[(i1 + 1) * n + (j1 + 1)]; register double c12 = C[(i1 + 1) * n + (j1 + 2)];
                                        register double c20 = C[(i1 + 2) * n + j1]; register double c21 = C[(i1 + 2) * n + (j1 + 1)]; register double c22 = C[(i1 + 2) * n + (j1 + 2)];

                                        for(k1 = k; k1 < ((n < k+b) ? n : k+b); k1+=3){
                                                register double a00 = A[i1 * n + k1]; register double a10 = A[(i1 + 1) * n + k1]; register double a20 = A[(i1 + 2) * n + k1];
                                                register double b00 = B[k1 * n + j1]; register double b01 = B[k1 * n + (j1 + 1)]; register double b02 = B[k1 * n + (j1 + 2)];

                                                c00 += a00*b00; c01 += a00*b01; c02 += a00*b02;
                                                c10 += a10*b00; c11 += a10*b01; c12 += a10*b02;
                                                c20 += a20*b00; c21 += a20*b01; c22 += a20*b02;

                                                //Similarly holding second column and row in A&B respectively
                                                a00 = A[i1 * n + k1 + 1]; a10 = A[(i1 + 1) * n + k1 + 1]; a20 = A[(i1 + 2) * n + k1 + 1];
                                                b00 = B[(k1 + 1) * n + j1]; b01 = B[(k1 + 1) * n + j1 + 1]; b02 = B[(k1 + 1) * n + j1 + 2];

                                                c00 += a00*b00; c01 += a00*b01; c02 += a00*b02;
                                                c10 += a10*b00; c11 += a10*b01; c12 += a10*b02;
                                                c20 += a20*b00; c21 += a20*b01; c22 += a20*b02;

                                                //Similarly holding third column and row in A&B respectively
                                                a00 = A[i1 * n + k1 + 2]; a10 = A[(i1 + 1) * n + k1 + 2]; a20 = A[(i1 + 2) * n + k1 + 2];
                                                b00 = B[(k1 + 2) * n + j1]; b01 = B[(k1 + 2) * n + j1 + 1]; b02 = B[(k1 + 2) * n + j1 + 2];

                                                c00 += a00*b00; c01 += a00*b01; c02 += a00*b02;
                                                c10 += a10*b00; c11 += a10*b01; c12 += a10*b02;
                                                c20 += a20*b00; c21 += a20*b01; c22 += a20*b02;
                                        }

                                        C[i1 * n + j1] = c00; C[i1 * n + (j1 + 1)] = c01; C[i1 * n + (j1 + 2)] = c02;
                                        C[(i1 + 1) * n + j1] = c10; C[(i1 + 1) * n + (j1 + 1)] = c11; C[(i1 + 1) * n + (j1 + 2)] = c12;
                                        C[(i1 + 2) * n + j1] = c20; C[(i1 + 2) * n + (j1 + 1)] = c21;  C[(i1 + 2) * n + (j1 + 2)] = c22;
                                }
                        }
                }
        }
}
}
