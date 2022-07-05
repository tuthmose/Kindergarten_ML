#include "stdio.h"

void reverse_pnt(float* var1, float* var2){
   float var3 = *var1;
   *var1 = *var2;
   *var2 = var3;
   printf("ByRef, Callee %p %p\n",var1,var2);
   }

void reverse_val(float var1, float var2){
   float var3 = var1;
   var1 = var2;
   var2 = var3;
   printf("ByVal, Callee %p %p\n",&var1,&var2);
   }   
   
void main(int argc, char** argv){
    float V1, V2;
    V1 = 1.0;
    V2 = 2.0;
    printf("Caller address %p %p\n",&V1,&V2);
    printf("Caller values %f %f\n",V1,V2);
    reverse_val(V1,V2);
    printf("Result %f %f\n",V1,V2);
    reverse_pnt(&V1,&V2);
    printf("Result %f %f\n",V1,V2);
    }