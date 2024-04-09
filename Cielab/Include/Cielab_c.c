/* C implementation

                  GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

 Copyright Yoann Berenguer

*/


#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>


struct im_stats{
    float red_mean;
    float red_std_dev;
    float green_mean;
    float green_std_dev;
    float blue_mean;
    float blue_std_dev;
};

struct lab{
    float l;
    float a;
    float b;
};

struct xyz{
    float x;
    float y;
    float z;
};


struct rgb{
    float r;
    float g;
    float b;
};


//
//int main(){
//return 0;
//}