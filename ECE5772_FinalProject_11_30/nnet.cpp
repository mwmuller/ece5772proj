
// ECE 5772: High Performance Embedded Programming
// Authors: Luke Nuculaj and Michael Muller
// The following is a parallel_pipeline implementation of a deep 
// neural network.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <tbb/tbb.h>
#include "nn_fun.h"
#include "FinishedNets/allNetworks.h"

#define EN_SEQ
#define EN_PARFOR
#define EN_PARPIP
#define EN_PARRED
#define NUM_RUNS (20)

#define ENABLE_BIG_NETWORK
#define I_CALLOC(x) (int*)calloc(x, sizeof(int)); 
#define D_CALLOC(x) (double*)calloc(x, sizeof(double)); 
#define MEMCPY(dest, orig, num) for(int cnt = 0; cnt < num; cnt++) dest[cnt] = orig[cnt];

using namespace tbb; 

void performBalance(int num_layers, int num_inputs, int num_outputs, 
                    int *layer_sizes, double * network_inputs,
                    const double *network_biases, const double *network_weights)
{
    double *a_in  = D_CALLOC(num_inputs);
    double *a_out = D_CALLOC(num_outputs);
    int sum_layer;
    int sum_out_layer; 
    NetworkParams params;
    LayerInfo info;
    struct timeval seq_start, seq_end, pf_start, pf_end, pp_start, pp_end, pr_start, pr_end;
    double seq_us = 0.0, pfor_us = 0.0, ppip_us = 0.0, pred_us = 0.0; 
    double seq_ans = 0.0, pfor_ans = 0.0, ppip_ans = 0.0, pred_ans = 0.0; // Maybe not needed

    params = 
        {
            num_layers,
            num_inputs,
            num_outputs,
            layer_sizes,
            network_biases,
            network_weights,
            network_inputs, 
            relu,
            purelin
        };  
        
    #ifdef EN_SEQ
    
        info = 
            {
                0,
                0,
                0,
                a_in,
                a_out,
            };
        
        /* FORWARD PROPAGATION */ 
        for(int y = 0; y < NUM_RUNS; y++)
        {
            gettimeofday(&seq_start, NULL);                                
            SequentialNeuralNet(&params, &info);
            gettimeofday(&seq_end, NULL); 
            
            // timing
            seq_us += (seq_end.tv_sec - seq_start.tv_sec)*1000000 + seq_end.tv_usec - seq_start.tv_usec;
        }
        printf("\n20 Run Avg Computation Time (sequential): %lf\n", seq_us/NUM_RUNS);
        printf("Output(s)\n"); 


        printOutput(info.a_out, num_outputs);

    #endif
    
    #ifdef EN_PARFOR

        info = 
            {
                0,
                0,
                0,
                a_in,
                a_out
            };

        for(int y = 0; y < NUM_RUNS; y++)
        {
            /* FORWARD PROPAGATION */   
            gettimeofday(&pf_start, NULL);                      
            ParForNeuralNet(&params, &info);
            gettimeofday(&pf_end, NULL);

            a_out[0] = info.a_out[0];
            // Without the above statement, we just pass the same value along
            pfor_us += (pf_end.tv_sec - pf_start.tv_sec)*1000000 + pf_end.tv_usec - pf_start.tv_usec;
        }
        printf("\n20 Run Avg Computation Time (parallel_for): %lf\n", pfor_us/NUM_RUNS);
        printf("Output(s)\n");

        printOutput(info.a_out, num_outputs);

    #endif
    
    #ifdef EN_PARPIP
    
        info = 
            {
                0,
                0,
                0,
                a_in,
                a_out
            };
        for(int y = 0; y < NUM_RUNS; y++)
        {
            // parallel pipelined implementation   
            gettimeofday(&pp_start, NULL);
            ParPipNeuralNet(&params, &info);
            gettimeofday(&pp_end, NULL); 
            
            a_out[0] = info.a_out[0];
            ppip_us += (pp_end.tv_sec - pp_start.tv_sec)*1000000 + pp_end.tv_usec - pp_start.tv_usec;
        }
        printf("\n20 Run Avg Computation Time (parallel_pipeline): %lf\n", ppip_us/NUM_RUNS);
        printf("Output(s)\n");

        printOutput(info.a_out, num_outputs);
    
    #endif

    #ifdef EN_PARRED
    
        info = 
            {
                0,
                0,
                0,
                a_in,
                a_out
            };

        for(int y = 0; y < NUM_RUNS; y++)
        {
            // parallel reduce implementation   
            gettimeofday(&pr_start, NULL);
            ParRedNeuralNet(&params, &info);  
            gettimeofday(&pr_end, NULL); 
            a_out[0] = info.a_out[0];
            pred_us += (pr_end.tv_sec - pr_start.tv_sec)*1000000 + pr_end.tv_usec - pr_start.tv_usec;
        }

        printf("\n20 Run Avg Computation Time (parallel_reduce): %lf\n", pred_us/NUM_RUNS);
        printf("Output(s)\n");

       printOutput(info.a_out, num_outputs);
    
    #endif
    
    free(a_in);
    free(a_out); 
    
}

int main(){
    
    /*
     *  INFO: 
     *  -----
     *  ENABLE_BIG_NETWORK loads and computes the output of a pre-trained neural net
     *  with 1 input, 1 output, and two hidden layers of size 5 and 5.
     *
     *  input: -1, expected result: -1.0082 (TEST PASSED)
     *  input:  0, expected result:  0.0020 (TEST PASSED)
     */
     
    /* PARAMETER INITIALIZATION */
    //3 Cell Pack
    int num_layers  = 6;
    int num_inputs  = 3;
    int num_outputs = 3;
    int layer_sizes_3[] = {3, 48, 48, 48, 48, 3}; // input - hidden layers - output
    // 3 Cell Pack
    double network_inputs_3[] = {0.8147, 0.9058, 0.1270};

    printf("\n**************3 Cell Pack Results***************\n");
    performBalance(num_layers,  num_inputs,  num_outputs, 
                     layer_sizes_3, network_inputs_3,
                    network_biases_3, network_weights_3);

    // 5 Cell Pack
    num_layers  = 6;
    num_inputs  = 5;
    num_outputs = 5;
    int layer_sizes_5[] = {5, 64, 64, 64, 64, 5}; // input - hidden layers - output
    
    // 5 Cell pack
    double network_inputs_5[] = {0.8147, 0.9058, 0.1270, 0.4, 0.1}; 
    printf("\n**************5 Cell Pack Results***************\n");
    performBalance(num_layers,  num_inputs,  num_outputs, 
                     layer_sizes_5, network_inputs_5,
                    network_biases_5, network_weights_5);

    //10 Cell Pack
    num_layers  = 6;
    num_inputs  = 10;
    num_outputs = 10;
    int layer_sizes_10[] = {10, 96, 128, 128, 96, 10}; // input - hidden layers - output

    // 10 Cell pack
    double network_inputs_10[] = {0.8147, 0.9058, 0.1270, 0.4, 0.1, 0.5, 0.6, 0.9, 0.1, 0.32}; 
    printf("\n**************10 Cell Pack Results***************\n");
    performBalance(num_layers,  num_inputs,  num_outputs, 
                     layer_sizes_10, network_inputs_10,
                    network_biases_10, network_weights_10);

    //20 Cell Pack
    num_layers  = 7;
    num_inputs  = 20;
    num_outputs = 20;
    int layer_sizes_20[] = {20, 256, 256, 256, 256, 256, 20}; // input - hidden layers - output

    // 20 Cell Pack
    double network_inputs_20[] = {
    0.8147, 0.9058, 0.1270, 0.4, 0.1, 
    0.8147, 0.9058, 0.1270, 0.4, 0.1, 
    0.8147, 0.9058, 0.1270, 0.4, 0.1, 
    0.8147, 0.9058, 0.1270, 0.4, 0.1};

    printf("\n**************20 Cell Pack Results***************\n");
    performBalance(num_layers,  num_inputs,  num_outputs, 
                     layer_sizes_20, network_inputs_20,
                    network_biases_20, network_weights_20);
    
    return 0;
}

