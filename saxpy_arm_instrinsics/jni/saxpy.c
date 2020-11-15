#include <arm_neon.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>
void calculate_saxpy(int a_constant, uint8x16_t *x_vector, uint8x16_t *y_vector, uint8x16_t *s_vector){
    //const uint8_t uint8_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    
    //Defining new vector for constant:
    uint8x16_t constant;
    //Replicate the a_constant in a vector:
    constant = vmovq_n_u8 (a_constant);

    //Defining new intern vector:
    uint8x16_t constant_times_x_vector;
    //a*x_vector:
    constant_times_x_vector = vmulq_u8(constant, *x_vector);

    //s_vector = a*x_vector + y_vector:
    *s_vector = vaddq_u8 (constant_times_x_vector, *y_vector);

}

void print_uint8 (uint8x16_t data, char* name) {
    int i;
    static uint8_t p[16];

    vst1q_u8 (p, data);

    printf ("%s = ", name);
    for (i = 0; i < 16; i++) {
	printf ("%02d ", p[i]);
    }
    printf ("\n");
}

void testSaxpy(){
//const uint8_t x_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }; //x
    //const uint8_t y_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }; //y
    //const uint8_t s_data[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0 };; //s
    uint8_t x_data[16000];
    uint8_t y_data[16000];
    uint8_t s_data[16000];

    uint8_t x_data_l[1600];
    uint8_t y_data_l[1600];
    uint8_t s_data_l[1600];

    uint8_t x_data_ll[160];
    uint8_t y_data_ll[160];
    uint8_t s_data_ll[160];

    for (int i = 0; i < 16000; i++)
    {
        //Building vectors with data:
        x_data[i] = i*2;
        y_data[i] = i*3;
        s_data[i] = 0;
    }

    for (int i = 0; i < 1600; i++)
    {
        //Building vectors with data:
        x_data_l[i] = i*2;
        y_data_l[i] = i*3;
        s_data_l[i] = 0;
    }

    for (int i = 0; i < 160; i++)
    {
        //Building vectors with data:
        x_data_ll[i] = i*2;
        y_data_ll[i] = i*3;
        s_data_ll[i] = 0;
    }

    
    //Taking a arbitrary constant:
    int a_constant = 2; //a

    //Init the clock variable:
    clock_t t;
    clock_t t2;
    clock_t t3;
    

    //Init the parallel saxpy calcultation:
    #pragma omp parallel
    #pragma omp single
    t = clock();

    #pragma omp for 
    for (int i = 0; i < 16000; i+=16)
    {
        //Convert slices of the main vectors in to smaller vectors:
        uint8_t x_data_inside[] = {x_data[i], x_data[i+1], x_data[i+2], x_data[i+3], x_data[i+4], x_data[i+5], x_data[i+6], x_data[i+7], x_data[i+8], x_data[i+9], x_data[i+10], x_data[i+11], x_data[i+12], x_data[i+13], x_data[i+14], x_data[i+15]};
        uint8_t y_data_inside[] = {y_data[i], y_data[i+1], y_data[i+2], y_data[i+3], y_data[i+4], y_data[i+5], y_data[i+6], y_data[i+7], y_data[i+8], y_data[i+9], y_data[i+10], y_data[i+11], y_data[i+12], y_data[i+13], y_data[i+14], y_data[i+15]};
        uint8_t s_data_inside[] = {s_data[i], s_data[i+1], s_data[i+2], s_data[i+3], s_data[i+4], s_data[i+5], s_data[i+6], s_data[i+7], s_data[i+8], s_data[i+9], s_data[i+10], s_data[i+11], s_data[i+12], s_data[i+13], s_data[i+14], s_data[i+15]};
        
        //Loading vectors into vectorial registers:
        uint8x16_t x_vector;
        x_vector = vld1q_u8 (x_data_inside);

        uint8x16_t y_vector;
        y_vector = vld1q_u8 (y_data_inside);

        uint8x16_t s_vector;
        s_vector = vld1q_u8 (s_data);

        print_uint8 (s_vector, "s_vector before saxpy");
        //Calculating saxpy for each slice:
        calculate_saxpy(a_constant, &x_vector, &y_vector, &s_vector);
        print_uint8 (s_vector, "s_vector after saxpy");
    }
    //Taking the ending time of the main process:
    #pragma omp single
    t = clock() - t;
    double tiempo_tomado = ((double)t)/CLOCKS_PER_SEC;
    printf("\033[1;31m El programa duró %f, con %i elementos.\033[0m; \n", tiempo_tomado, 16000);

    //Assigning the second time:
    t2 = clock();

    #pragma omp for 
    for (int i = 0; i < 1600; i+=16)
    {
        //Convert slices of the main vectors in to smaller vectors:
        uint8_t x_data_inside[] = {x_data_l[i], x_data_l[i+1], x_data_l[i+2], x_data_l[i+3], x_data_l[i+4], x_data_l[i+5], x_data_l[i+6], x_data_l[i+7], x_data_l[i+8], x_data_l[i+9], x_data_l[i+10], x_data_l[i+11], x_data_l[i+12], x_data_l[i+13], x_data_l[i+14], x_data_l[i+15]};
        uint8_t y_data_inside[] = {y_data_l[i], y_data_l[i+1], y_data_l[i+2], y_data_l[i+3], y_data_l[i+4], y_data_l[i+5], y_data_l[i+6], y_data_l[i+7], y_data_l[i+8], y_data_l[i+9], y_data_l[i+10], y_data_l[i+11], y_data_l[i+12], y_data_l[i+13], y_data_l[i+14], y_data_l[i+15]};
        uint8_t s_data_inside[] = {s_data_l[i], s_data_l[i+1], s_data_l[i+2], s_data_l[i+3], s_data_l[i+4], s_data_l[i+5], s_data_l[i+6], s_data_l[i+7], s_data_l[i+8], s_data_l[i+9], s_data_l[i+10], s_data_l[i+11], s_data_l[i+12], s_data_l[i+13], s_data_l[i+14], s_data_l[i+15]};
        
        //Loading vectors into vectorial registers:
        uint8x16_t x_vector;
        x_vector = vld1q_u8 (x_data_inside);

        uint8x16_t y_vector;
        y_vector = vld1q_u8 (y_data_inside);

        uint8x16_t s_vector;
        s_vector = vld1q_u8 (s_data);

        print_uint8 (s_vector, "s_vector before saxpy");
        //Calculating saxpy for each slice:
        calculate_saxpy(a_constant, &x_vector, &y_vector, &s_vector);
        print_uint8 (s_vector, "s_vector after saxpy");
    }
    #pragma omp single
    t2 = clock() - t2;
    double tiempo_tomado2 = ((double)t2)/CLOCKS_PER_SEC;
    printf("\033[1;31m El programa duró %f, con %i elementos.\033[0m; \n", tiempo_tomado2, 1600);
    
    t3 = clock() -t3;

    #pragma omp for 
    for (int i = 0; i < 160; i+=16)
    {
        //Convert slices of the main vectors in to smaller vectors:
        uint8_t x_data_inside[] = {x_data_ll[i], x_data_ll[i+1], x_data_ll[i+2], x_data_ll[i+3], x_data_ll[i+4], x_data_ll[i+5], x_data_ll[i+6], x_data_ll[i+7], x_data_ll[i+8], x_data_ll[i+9], x_data_ll[i+10], x_data_ll[i+11], x_data_ll[i+12], x_data_ll[i+13], x_data_ll[i+14], x_data_ll[i+15]};
        uint8_t y_data_inside[] = {y_data_ll[i], y_data_ll[i+1], y_data_ll[i+2], y_data_ll[i+3], y_data_ll[i+4], y_data_ll[i+5], y_data_ll[i+6], y_data_ll[i+7], y_data_ll[i+8], y_data_ll[i+9], y_data_ll[i+10], y_data_ll[i+11], y_data_ll[i+12], y_data_ll[i+13], y_data_ll[i+14], y_data_ll[i+15]};
        uint8_t s_data_inside[] = {s_data_ll[i], s_data_ll[i+1], s_data_ll[i+2], s_data_ll[i+3], s_data_ll[i+4], s_data_ll[i+5], s_data_ll[i+6], s_data_ll[i+7], s_data_ll[i+8], s_data_ll[i+9], s_data_ll[i+10], s_data_ll[i+11], s_data_ll[i+12], s_data_ll[i+13], s_data_ll[i+14], s_data_ll[i+15]};
        
        //Loading vectors into vectorial registers:
        uint8x16_t x_vector;
        x_vector = vld1q_u8 (x_data_inside);

        uint8x16_t y_vector;
        y_vector = vld1q_u8 (y_data_inside);

        uint8x16_t s_vector;
        s_vector = vld1q_u8 (s_data);

        print_uint8 (s_vector, "s_vector before saxpy");
        //Calculating saxpy for each slice:
        calculate_saxpy(a_constant, &x_vector, &y_vector, &s_vector);
        print_uint8 (s_vector, "s_vector after saxpy");
    }

    #pragma omp single
    t3 = clock() - t3;
    double tiempo_tomado3 = ((double)t3)/CLOCKS_PER_SEC;
    printf("\033[1;31m El programa duró %f, con %i elementos.\033[0m; \n", tiempo_tomado3, 160);
}

double evaluateFunction( double  x){
    return (double ) 4*x*x*x + sin((double )x)*cos((double )x)-x*x;

}


double calculateIntegral(int subdivisions, int initX,int finalX){
    double start_time = omp_get_wtime();

    double sum = 0;
    double differential = (double ) (finalX-initX)/subdivisions;
#pragma omp parallel
    {
#pragma omp for reduction(+:sum)
        for (int i = initX; i < finalX; ++i) {
            double resum = (double)i;
            while (resum<i+1){
                sum+= differential*evaluateFunction((double ) (resum+(resum+differential)/2));
                resum+=differential;
            }
        }
    }
    double finishtime = omp_get_wtime() - start_time;
    printf(" La sumatoria tardo aproximadamente %lf con un resultado de %lf",finishtime,sum);
    return sum;
}
void euler_serial(int iterations) {
    double start_time = omp_get_wtime();
    double e;
    double * a = malloc(sizeof(double )*iterations);
    a[0] = 1.0;
    for (int i = 1; i < iterations; i++)
    {
        a[i] = a[i-1] / i;
    }
    e = 1.;
    for (int i = iterations - 1; i > 0; i--)
        e += a[i];

    double finishtime = omp_get_wtime() - start_time;
    printf("Se tardo: %lf", finishtime);
    printf("Euler constant e = %.16lf\n", e);
    free(a);
}
void euler_parallel(int iterations) {
    double start_time = omp_get_wtime();
    double e;
    double * a = malloc(sizeof(double )*iterations);
    e = 1.;
    a[0] = 1.0;
#pragma omp parallel
    {
#pragma omp for
        for (int i = 1; i < iterations; i++)
        {
            a[i] = a[i-1] / i;
        }

#pragma omp for reduction(+:e)
        for (int i = iterations - 1; i > 0; i--)
            e += a[i];
    }
    double finishtime = omp_get_wtime() - start_time;
    printf("Se tardo: %lf", finishtime);
    printf("Euler constant e = %.16lf\n", e);
    free(a);
}

int main(int argc, char* argv[]){
    //__________Pruebas generales, descomentar para ejecutarlas_______________//
    // omp_set_num_threads(6);
    // printf("_______Saxpy test______\n");
    // testSaxpy();
    // printf("\n _______Saxpy test Final______\n");
    // printf("_______Init euler serial________\n");
    // euler_serial(100000);
    // printf("\n _______Fin euler serial______\n");
    // printf("_______Init euler parallel_______\n");
    // euler_parallel(100000);
    // printf("\n _______Fin euler parallel_______\n");
    // printf("_______Init calculo integral_______\n");
    // calculateIntegral(1000000000,1,100);
    // printf("\n _______Fin calculo integral_______\n");
    // printf("_______Init calculo integral serial_______\n");
    // omp_set_num_threads(1);
    // calculateIntegral(1000000000,1,100);
    // printf("\n_______Fin calculo integral serial_______\n");

    //__________Pruebas de euler tanto serial como paralelo, descomentar para ejecutarlas_______________//
    // printf("\nEjecucion en serial 1000000\n");
    // euler_serial(1000000);
    // printf("\n 2000000\n");
    // euler_serial(2000000);
    // printf("\n3000000 \n");
    // euler_serial(3000000);
    // printf("\n 4000000 \n");
    // euler_serial(4000000);
    // printf("\n 5000000 \n");
    // euler_serial(5000000);
    // printf("\n6000000 \n");
    // euler_serial(6000000);
    // printf("\n Paralelo 1000000 \n");
    // euler_parallel(1000000);
    // printf("\n 2000000\n");
    // euler_parallel(2000000);
    // printf("\n3000000 \n");
    // euler_parallel(3000000);
    // printf("\n 4000000 \n");
    // euler_parallel(4000000);
    // printf("\n 5000000 \n");
    // euler_parallel(5000000);
    // printf("\n6000000 \n");
    // euler_parallel(600000);

    //__________Pruebas de integral tanto serial como paralelo, descomentar para ejecutarlas_______________//

printf("\n Ejecucion de calculo de integral 1 hilo(serial)\n");
    omp_set_num_threads(1);
calculateIntegral(1000000000,1,100);
printf("\n Ejecucion de calculo de integral 2 hilos\n");

    omp_set_num_threads(2);
calculateIntegral(1000000000,1,100);
printf("\n Ejecucion de calculo de integral 3 hilos\n");
    omp_set_num_threads(3);
    calculateIntegral(1000000000,1,100);
printf("\n Ejecucion de calculo de integral 4 hilos\n");
    omp_set_num_threads(4);
    calculateIntegral(1000000000,1,100);
printf("\n Ejecucion de calculo de integral 5 hilos\n");


    omp_set_num_threads(5);
calculateIntegral(1000000000,1,100);
printf("\n Ejecucion de calculo de integral 6 hilos\n");

    omp_set_num_threads(6);
calculateIntegral(1000000000,1,100);

}
