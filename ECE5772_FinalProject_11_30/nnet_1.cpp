
// ECE 5772: High Performance Embedded Programming
// Authors: Luke Nuculaj and Michael Muller
// The following is a parallel_pipeline implementation of a deep 
// neural network.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <tbb/tbb.h>
#include "nn_fun_1.h"
#include "FinishedNets/allNetworks.h"

//#define EN_SEQ
//#define EN_PARFOR
//#define EN_PARPIP
#define EN_PARRED

#define ENABLE_BIG_NETWORK
#define I_CALLOC(x) (int*)calloc(x, sizeof(int)); 
#define D_CALLOC(x) (double*)calloc(x, sizeof(double)); 
#define MEMCPY(dest, orig, num) for(int cnt = 0; cnt < num; cnt++) dest[cnt] = orig[cnt];

using namespace tbb; 

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
    int num_cells   = 20; 
    int num_layers  = 7;
    int num_inputs  = 20;
    int num_outputs = 20;
    int layer_sizes[] = {20, 256, 256, 256, 256, 256, 20}; // input - hidden layers - output
    
    /*
     double network_biases[] = 
            {
               0.6174, -0.9363, -0.4462, -0.9077, -0.8057,
               -1.0392, -1.2578, -0.3869, -0.0865, 1.3236,
               2.2620
            };
     double network_weights[] = 
            {   
               2.4611, 0.4863, -0.2155, 0.3110, -0.6576,
               
               1.9185, -0.1225, -0.0205, -0.4479, -0.0033,
               -1.2188, -0.2369, -0.1088, 0.3594, 0.9195,
               -0.3734, 0.5310, 0.2926, 0.3102, -0.3192,
               2.0691, 0.5904, 0.4187, -0.6748, 0.1705,
               -1.8475, -0.6263, 0.5094, -0.7620, -0.5524,
               
               2.0718, -2.5593, -0.7200, -1.7937, -2.3317
            };
    */
    
    double *state_of_charge = D_CALLOC(num_cells);
    FILE *fptr;
    
    if (num_cells == 3){
        double init[] = {0.5508, 0.7081, 0.2909};
        MEMCPY(state_of_charge, init, num_cells);
        fptr = fopen("results_3_cell.txt", "w");
    }
    else if (num_cells == 5){
        double init[] = {0.5508, 0.7081, 0.2909, 0.5108, 0.8929};
        MEMCPY(state_of_charge, init, num_cells);
        fptr = fopen("results_5_cell.txt", "w");
    }
    else if (num_cells == 10){
        double init[] = {0.5508, 0.7081, 0.2909, 0.5108, 0.8929,
                         0.8963, 0.1256, 0.2072, 0.0515, 0.4408};
        MEMCPY(state_of_charge, init, num_cells);
        fptr = fopen("results_10_cell.txt", "w");
    }
    else if (num_cells == 20){
        double init[] = {0.5508, 0.7081, 0.2909, 0.5108, 0.8929,
                         0.8963, 0.1256, 0.2072, 0.0515, 0.4408,
                         0.0299, 0.4568, 0.6491, 0.2785, 0.6763,
                         0.5909, 0.0240, 0.5589, 0.2593, 0.4151};
        MEMCPY(state_of_charge, init, num_cells);
        fptr = fopen("results_20_cell.txt", "w");
    }       
    //double network_inputs[] = {0.8147, 0.9035, 0.1292, 0.9111, 0.6346}; 
    
    
    int num_samples = 1000;
    for(int i=0; i<num_samples; i++){
    
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
                network_biases_20,
                network_weights_20,
                state_of_charge, 
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
            gettimeofday(&seq_start, NULL);                                
            SequentialNeuralNet(&params, &info);
            gettimeofday(&seq_end, NULL); 
            
            // timing
            seq_us = (seq_end.tv_sec - seq_start.tv_sec)*1000000 + seq_end.tv_usec - seq_start.tv_usec;
            printf("\nComputation Time (sequential): %lf\n", seq_us);
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
            /* FORWARD PROPAGATION */   
            gettimeofday(&pf_start, NULL);                      
            ParForNeuralNet(&params, &info);
            gettimeofday(&pf_end, NULL);

            a_out[0] = info.a_out[0];
            // Without the above statement, we just pass the same value along
            pfor_us = (pf_end.tv_sec - pf_start.tv_sec)*1000000 + pf_end.tv_usec - pf_start.tv_usec;
            printf("\nComputation Time (parallel_for): %lf\n", pfor_us);
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
            
            // parallel pipelined implementation   
            gettimeofday(&pp_start, NULL);
            ParPipNeuralNet(&params, &info);
            gettimeofday(&pp_end, NULL); 
            
            a_out[0] = info.a_out[0];
            ppip_us = (pp_end.tv_sec - pp_start.tv_sec)*1000000 + pp_end.tv_usec - pp_start.tv_usec;
            printf("\nComputation Time (parallel_pipeline): %lf\n", ppip_us);
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
            // parallel reduce implementation   
            gettimeofday(&pr_start, NULL);
            ParRedNeuralNet(&params, &info);  
            gettimeofday(&pr_end, NULL); 
            a_out[0] = info.a_out[0];
            pred_us = (pr_end.tv_sec - pr_start.tv_sec)*1000000 + pr_end.tv_usec - pr_start.tv_usec;
            //printf("\nComputation Time (parallel_reduce): %lf\n", pred_us);
            //printf("Output(s)\n");
            
            //printOutput(info.a_out, num_outputs);
            updateBatteryPack(state_of_charge, info.a_out, num_cells);
            printf("Sample number %d\n", i);
            for(int cnt=0; cnt<num_cells; cnt++){
                printf("x[%d] = %lf\n", cnt, state_of_charge[cnt]);
                fprintf(fptr, "%0.5lf\n", state_of_charge[cnt]);
            }
            for(int cnt=0; cnt<num_cells; cnt++){
                fprintf(fptr, "%0.5lf\n", info.a_out[cnt]);
            }
        #endif
        
        free(a_in);
        free(a_out); 
    }
    
    fclose(fptr);
    return 0;
}
