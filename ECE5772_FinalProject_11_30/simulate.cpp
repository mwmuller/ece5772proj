
// ECE 5772: High Performance Embedded Programming
// Authors: Luke Nuculaj and Michael Muller
// The following is a parallel_pipeline implementation of a deep 
// neural network.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <tbb/tbb.h>
#include "simulate_fun.h"
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

int main(int argc, char** argv){
    
    /*
     *  INFO: 
     *  -----
     *  ENABLE_BIG_NETWORK loads and computes the output of a pre-trained neural net
     *  with 1 input, 1 output, and two hidden layers of size 5 and 5.
     *
     *  input: -1, expected result: -1.0082 (TEST PASSED)
     *  input:  0, expected result:  0.0020 (TEST PASSED)
     */
    int num_cells;
    int sampleRate; 
    if (argc < 2)
    {
	printf("Defaulting num cells to 3\n");
	num_cells = 3;
	sampleRate = 1;
	
    }else{
	num_cells = atoi(argv[1]);
	if (argc < 3) sampleRate = 1; // default the sample rate to 1
	else sampleRate = atoi(argv[2]);
	printf("Number of cells selected: %d\n", num_cells);
    }
    /* PARAMETER INITIALIZATION */
    int num_layers;
    int num_inputs;
    int num_outputs;
    int layer_sizes_5[] = {5, 64, 64, 64, 64, 5};
    int layer_sizes_3[] = {3, 48, 48, 48, 48, 3}; // input - hidden layers - output
    int layer_sizes_10[] = {10, 96, 128, 128, 96, 10};
    int layer_sizes_20[] = {20, 256, 256, 256, 256, 256, 20};
    int *layer_sizes;
    const double *network_biases;
    const double *network_weights;
    double *state_of_charge = D_CALLOC(num_cells);
    FILE *fptr;
    
    if (num_cells == 3){
	num_layers  = 6;
	num_inputs  = 3;
	num_outputs = 3;
	layer_sizes = layer_sizes_3;
        double init[] = {0.5508, 0.7081, 0.2909};
	network_biases = network_biases_3;
        network_weights = network_weights_3;
        MEMCPY(state_of_charge, init, num_cells);
        fptr = fopen("results_3_cell.txt", "w");
    }
    else if (num_cells == 5){

	num_layers  = 6;
	num_inputs  = 5;
	num_outputs = 5;
	layer_sizes = layer_sizes_5;
	network_biases = network_biases_5;
        network_weights = network_weights_5;

        double init[] = {0.5508, 0.7081, 0.2909, 0.5108, 0.8929};

        MEMCPY(state_of_charge, init, num_cells);
        fptr = fopen("results_5_cell.txt", "w");
    }
    else if (num_cells == 10){

	num_layers  = 6;
	num_inputs  = 10;
	num_outputs = 10;
	layer_sizes = layer_sizes_10;
	network_biases = network_biases_10;
        network_weights = network_weights_10;

        double init[] = {0.5508, 0.7081, 0.2909, 0.5108, 0.8929,
                         0.8963, 0.1256, 0.2072, 0.0515, 0.4408};

        MEMCPY(state_of_charge, init, num_cells);
        fptr = fopen("results_10_cell.txt", "w");
    }
    else if (num_cells == 20){

	num_layers  = 7;
	num_inputs  = 20;
	num_outputs = 20;
	
	layer_sizes = layer_sizes_20;
	network_biases = network_biases_20;
        network_weights = network_weights_20;

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
                network_biases,
                network_weights,
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

	if (i % sampleRate == 0)
	{
            printf("Sample number %d\n", i);
	    fprintf(fptr, "Sample Number: %d\n", i);
            for(int cnt=0; cnt<num_cells; cnt++){
		printf("x[%d] = %lf\n", cnt, state_of_charge[cnt]);
                fprintf(fptr, "x[%d] = %0.5lf | u[%d] = %0.5lf\n", cnt, state_of_charge[cnt], cnt, info.a_out[cnt]);
		            
	    }
	}
        #endif
        
        free(a_in);
        free(a_out); 
    }
    
    fclose(fptr);
    return 0;
}
