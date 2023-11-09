#include <stdio.h>
#include <math.h>
#include <vector>
#include <utility>
#include <iostream>
#include <algorithm>
#include <iterator>

// Physical constraints
#define BETA (0.000675f) // assume same beta for all cells
#define UMAX (0.3f) // max balancing current to be applied
#define SIZE (3) // size of pack
#define STEPt (20) // how often we check the battery SOCs to apply balancing
// defined environment parameters
#define SAFETHRESH (0.90f) // threshold before we throttle balancing current
#define EXTTIME (2000) // maximum additional time for balancing
#define UPPERSOC (1.015f) // Check if above the 
#define LOWERSOC (0.985f)
#define THROTTLESCALAR (1.263) // how much to we increase the throttle? (current best 1.263)


float getMaxDiff(float *arr, int n, float xBal, int maxDiffCount, bool &isMaxUpper)
{
    float max = 0;
    float diffCalc =0;
    for(int i = 0; i < n; i++)
    {
        // provide proper threshold
        if(arr[i] > xBal)
        {
            isMaxUpper = true;
        }
        diffCalc = abs(xBal - arr[i]);
        if(diffCalc > max)
        {
            max = diffCalc;
        }
        else if(diffCalc == max)
        {
            // find how many, and store the value of the pack
            maxDiffCount++;
        }
    }
    return max;
}

float getXBalanced(float *arr, int n)
{
    float sum = 0;
    for(int i = 0; i < n; i++)
    {
        sum+= arr[i];
    }
    return (sum/static_cast<float>(n));
}

float calcTime(float inUMax, float maxDiff, int count, bool isMaxUpper, int &throttleTime)
{
    // is the max diff coming from an above charged cell?
    float tempDiff = maxDiff;
    if(isMaxUpper)
    {
        maxDiff -= (maxDiff*(1-SAFETHRESH));
    }
    else
    {
        maxDiff = maxDiff*SAFETHRESH;
    }
    throttleTime = static_cast<int>(abs(tempDiff-maxDiff)/(BETA*(inUMax/static_cast<float>(count+1)))+1);
    printf("Full charge time required if linear discharge is: %d\n", throttleTime);
    return ((maxDiff)/(BETA*(inUMax/static_cast<float>(count+1))));
}

float calcXu(float xBalance, float Xi, int time, bool isAboveBalance)
{
    float diff = xBalance - Xi;
    if(isAboveBalance)
    {
        diff -= (diff*(1-SAFETHRESH));
    }
    else
    {
        diff = diff*SAFETHRESH;
    }
    return (diff)/(static_cast<float>(time) * BETA);
}

void getLowerUpperSoc(float *arr, int n, float xBal, std::vector<std::pair<float, int>> &xLower, std::vector<std::pair<float, int>> &xUpper, int maxIdx)
{
    // search the array and get the values lower than xbal
    std::pair<float, int> localP;
    for(int x = 0; x < n; x++)
    {
        // assign the lowers
        localP.first = arr[x];
        localP.second = x;
        if(arr[x] < xBal)
        {
            xLower.push_back(localP);
        }
        else if(arr[x] > xBal)
        {
            xUpper.push_back(localP);
        }
}
}

void calcUVecter(int _time, float *uVector, float xBalance, float *pack, std::vector<std::pair<float, int>> &upper, std::vector<std::pair<float, int>> &lower, bool isMaxUpper)
{
    // [Xu * 0.000675] * _Time = Xbal[i] - X[i]
    // Xu = [Xbal - X[i]]/(_Time * 0.000675)

    for(const auto& pair : upper)
    {
        uVector[pair.second] = calcXu(xBalance, pack[pair.second], _time, true);
        printf("pair first %f | pair second %d Xbalance %f, time = %d\n", pair.first, pair.second, xBalance, _time);
    }
    for(const auto& pair : lower)
    {
        uVector[pair.second] = calcXu(xBalance, pack[pair.second], _time, false);
        printf("pair first %f | pair second %d Xbalance %f, time = %d\n", pair.first, pair.second, xBalance, _time);
    }
    for(int t = 0; t < SIZE; t++)
    {
        printf("pack[%d]: %f U[%d]: %f\n",t, pack[t], t, uVector[t]);
    }
    printf("\n");
}

void executeEnv(float *pack, float *uVector, int packSize, int _Time, float *statePack, float *stateUVector, float xBal, int throttleTime)
{
    // assume each iteration is 1 second

    for(int t = 0; t < _Time; t++)
    { // execute multiplcation x times
        for(int x = 0; x < packSize; x++)
        {
            statePack[x] += (BETA*uVector[x]);
        }
        if(t % 500 == 0)
        {
            printf("**********\n");
            for(int d = 0; d < packSize; d++)
            {
                printf("StatePack[%d]: %f  State uVector[%d]: %f \n", d, statePack[d], d, stateUVector[d]);
            }
            printf("Timing %d\n", _Time);
        }
    }
    printf("checking extended=============================\n");
    // now determine when we hit our max
    for(int y = 0; y < EXTTIME; y+=STEPt)
    {
        for(int a = 0; a < packSize; a++)
        {
            stateUVector[a] -= (stateUVector[a] / (throttleTime))*THROTTLESCALAR*STEPt; // reduce current balance by some value calculated
            
            statePack[a] += (BETA*stateUVector[a]*STEPt);
        }
        if(y % 500 == 0)
        {
            printf("**********\n");
            for(int d = 0; d < packSize; d++)
            {
                printf("StatePack[%d]: %f  State uVector[%d]: %f \n", d, statePack[d], d, stateUVector[d]);
            }
            printf("Timing %d\n", _Time);
        }
        if(statePack[0] < (xBal*UPPERSOC) && statePack[0] > (xBal*LOWERSOC))
        {
            int totalTime = (_Time + y);
            printf("balance range upper %f\n", xBal*UPPERSOC);
            printf("balance range lower %f\n", xBal*LOWERSOC);
            printf("Time before throttle: %d\n", _Time);
            printf("Total time to max was: %d\n", totalTime);
            printf("Total Throttle time: %d\n", totalTime - _Time);
            
            break;
        }
    }
}

// input size of vector, uMax, 
int main(int argc, char **argv)
{
    //float pack[SIZE] = {0.5, 0.4, 0.6}; // [0, 0.3005, -0.3005]
    //float pack[SIZE] = {1.0, 0.4, 0.6}; // [-.3, 0.2400, 0.06]
    //float pack[SIZE] = {0.6, 0.6, 0.4 ,0.4}; // [-.3, -.3, .3 .3]
    //float pack[SIZE] = {1.0, 0.1, 0.26, 0.6}; // [-0.300062, 0.229459, 0.135322, -0.064719]
    
    int inSIZE;
    float inUMax;

    if(argc == 3)
    {
        // attempt to read
        inSIZE = atoi(argv[1]);
        inUMax = static_cast<float>(atoi(argv[2]))/10;
        printf("insize = %d\n", inSIZE);
        printf("inUmax = %f\n", inUMax);
    }
    else
    {
        inSIZE = SIZE;
        inUMax = UMAX;
    }

    float *pack = (float *) calloc (sizeof(float), inSIZE);
    for(int h = 0; h < inSIZE; h++)
    {
        pack[h] = static_cast<float>(rand() % 1000) / 1000;
    }
    float *PackDiff = (float *) calloc(sizeof(float), inSIZE);
    float *uVector = (float *) calloc (sizeof(float), inSIZE);
    float *statePack = (float *) calloc (sizeof(float), inSIZE); // track updating states
    float *stateUVector = (float *) calloc (sizeof(float), inSIZE);

    std::vector<std::pair<float, int>> upper;
    std::vector<std::pair<float, int>> lower;
    float xBalanced=0;
    float maxDiff =0;
    int throttleTime =0;
    bool isMaxUpper = false; // is the max blaanced diff from an upper or lower charged cell
    int maxIdx = 0; // find max diffs at all instances in an array
    int _Time = 0;
    int maxDiffCount =0; // stores number of tied max SOCs for time caluclation


    // SAFETHRESH (0.90f) // 
    // EXTTIME (2000) // maxi
    // UPPERSOC (1.015f) // C
    // LOWERSOC (0.985f)
    // THROTTLESCALAR (1.263)

    xBalanced = getXBalanced(pack, inSIZE);
    maxDiff = getMaxDiff(pack, inSIZE, xBalanced, maxDiffCount, isMaxUpper);
    _Time = calcTime(inUMax, maxDiff, maxDiffCount, isMaxUpper, throttleTime);
    getLowerUpperSoc(pack, inSIZE, xBalanced, lower, upper, maxIdx);
    
    printf("Max bal Diff: %f\n", maxDiff);
    printf("xBalanced Values: %f\n", xBalanced);
    printf("Calced Time min: %d with safe thresh of %f\n", _Time, SAFETHRESH);
    printf("MaxBalance Idx %d\n", maxIdx);
    // calc needs for packs below SOC
    calcUVecter(_Time, uVector, xBalanced, pack, upper, lower, isMaxUpper);
    float socSum = 0;
    for(int y = 0; y < inSIZE; y++)
    {
        socSum+= uVector[y];
    }
    printf("Total SOC sum %f\n", socSum);

    // create statePack
    for(int t = 0; t < inSIZE; t++)
    {
        statePack[t] = pack[t];
        stateUVector[t] = uVector[t];
    }
    executeEnv(pack, uVector, inSIZE, _Time, statePack, stateUVector, xBalanced, throttleTime);
    for(int t = 0; t < inSIZE; t++)
    {
        printf("pack[%d]: %f U[%d]: %f sPack[%d]: %f \n",t, pack[t], t, uVector[t], t, statePack[t]);
    }
    return 0;
}