/*******************************************************************************
*    nn.c   1.0                                       � JOHN BULLINARIA  2004  *
*******************************************************************************/

/*      To compile use "cc nn.c -O -lm -o nn" and then run using "./nn"       */
/*      For explanations see:  http://www.cs.bham.ac.uk/~jxb/INC/nn.html      */

#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>

#define NUMPAT 4
#define NUMIN  2
#define NUMHID 2
#define NUMOUT 1

#define rando() ((double)rand()/((double)RAND_MAX+1))

#if DBG == 0
#define dbg(...) ((void*)0)
#else
#define dbg(...) fprintf(stdout, __VA_ARGS__)
#endif

#if !defined(EPOCH_COUNT) && (DBG == 1)
#define EPOCH_COUNT 1
#endif

#if !defined(EPOCH_COUNT) && (DBG == 0)
#define EPOCH_COUNT 100000
#endif

int main(void) {
    int    i, j, k, p, np, op, ranpat[NUMPAT+1], epoch;
    int    NumPattern = NUMPAT, NumInput = NUMIN, NumHidden = NUMHID, NumOutput = NUMOUT;
    double Input[NUMPAT+1][NUMIN+1] = { {0, 0, 0},  {0, 0, 0},  {0, 1, 0},  {0, 0, 1},  {0, 1, 1} };
    double Target[NUMPAT+1][NUMOUT+1] = { {0, 0},  {0, 0},  {0, 1},  {0, 1},  {0, 0} };
    double SumH[NUMPAT+1][NUMHID+1], WeightIH[NUMIN+1][NUMHID+1], Hidden[NUMPAT+1][NUMHID+1];
    double SumO[NUMPAT+1][NUMOUT+1], WeightHO[NUMHID+1][NUMOUT+1], Output[NUMPAT+1][NUMOUT+1];
    double DeltaO[NUMOUT+1], SumDOW[NUMHID+1], DeltaH[NUMHID+1];
    double DeltaWeightIH[NUMIN+1][NUMHID+1], DeltaWeightHO[NUMHID+1][NUMOUT+1];
    double Error, eta = 0.5, alpha = 0.9, smallwt = 0.5;

    srand(1) ; // Initialize rand so results are consistent.
  
    for( j = 1 ; j <= NumHidden ; j++ ) {    /* initialize WeightIH and DeltaWeightIH */
        for( i = 0 ; i <= NumInput ; i++ ) { 
            DeltaWeightIH[i][j] = 0.0 ;
            WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
            dbg("WeightIH[%d][%d]=%lf\n", i, j, WeightIH[i][j]);
        }
    }
    for( k = 1 ; k <= NumOutput ; k ++ ) {    /* initialize WeightHO and DeltaWeightHO */
        for( j = 0 ; j <= NumHidden ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;              
            WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
            dbg("WeightHO[%d][%d]=%lf\n", j, k, WeightHO[j][k]);
        }
    }
     
    int epoch_count = EPOCH_COUNT ;
    for( epoch = 0 ; epoch < epoch_count ; epoch++) {    /* iterate weight updates */
        for( p = 1 ; p <= NumPattern ; p++ ) {    /* randomize order of training patterns */
            ranpat[p] = p ;
        }

        for( p = 1 ; p <= NumPattern ; p++) {
            double ro = rando();
            np = p + ro * ( NumPattern + 1 - p ) ;
            op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
            dbg("ro=%lf np=%d ranpat[%d]=%d\n", ro, np, p, ranpat[p]);
        }

        Error = 0.0 ;
        double sse, err ;
        for( np = 1 ; np <= NumPattern ; np++ ) {    /* repeat for all the training patterns */
            p = ranpat[np];
            for ( i = 1 ; i <= NumInput ; i++ ) {
                dbg("Input[%d][%d]=%lf\n", p, i, Input[p][i]);
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit activations */
                SumH[p][j] = WeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) {
                    SumH[p][j] += Input[p][i] * WeightIH[i][j] ;
                }
                Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
                dbg("Hidden[%d][%d]=%lf SumH[%d][%d]=%lf\n", p, j, Hidden[p][j], p, j, SumH[p][j]);
            }
            for( k = 1 ; k <= NumOutput ; k++ ) {    /* compute output unit activations and errors */
                SumO[p][k] = WeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
                }
                Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   /* Sigmoidal Outputs */
                dbg("%d:%d Target=%lf Output=%lf SumO=%lf\n",
                    p, k, Target[p][k], Output[p][k], SumO[p][k]);
/*              Output[p][k] = SumO[p][k];      Linear Outputs */
                err = Target[p][k] - Output[p][k] ;
                sse = 0.5 * err * err ;   /* SSE */
                Error += sse;
/*              Error -= ( Target[p][k] * log( Output[p][k] ) + ( 1.0 - Target[p][k] ) * log( 1.0 - Output[p][k] ) ) ;    Cross-Entropy Error */
                DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   /* Sigmoidal Outputs, SSE */
/*              DeltaO[k] = Target[p][k] - Output[p][k];     Sigmoidal Outputs, Cross-Entropy Error */
/*              DeltaO[k] = Target[p][k] - Output[p][k];     Linear Outputs, SSE */
                dbg("%d:%d err=%lf, DeltaO[%d]=%lf sse=%lf\n", p, k, err, k, DeltaO[k], sse);
            }
            dbg("back-propagate errors:\n");
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                SumDOW[j] = 0.0 ;
                for( k = 1 ; k <= NumOutput ; k++ ) {
                    SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
                    dbg("SumDOW[%d]:%lf += WeightHO[%d][%d]:%lf * DeltaO[%d]:%lf\n",
                        j, SumDOW[j], j, k, WeightHO[j][k], k, DeltaO[k]);
                }
                DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
                dbg("DeltaH[%d]:%lf = SumDOW[%d]:%lf * Hidden[%d][%d]:%lf * (1.0 - Hidden[%d][%d]:%lf\n",
                     j, DeltaH[j], j, SumDOW[j], p, j, Hidden[p][j], p, j, Hidden[p][j]);
            }
            dbg("Update WeightsIH: eta=%lf alpha=%lf\n", eta, alpha);
            for( j = 1 ; j <= NumHidden ; j++ ) {     /* update weights WeightIH */
                double momentum;
                double w;
#if DBG == 0
                DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j] ;
                WeightIH[0][j] +=  DeltaWeightIH[0][j] ;
#else
                momentum = alpha * DeltaWeightIH[0][j] ;
                DeltaWeightIH[0][j] = (eta * DeltaH[j]) + momentum ;
                dbg("DeltaWeightIH[%d][%d]:%lf = (eta:%lf * DeltaH[%d]:%lf) + momentum:%lf BIAS\n",
                     0, j, DeltaWeightIH[0][j], eta, j, DeltaH[j], momentum);
                w = WeightIH[0][j];
                WeightIH[0][j] = WeightIH[0][j] + DeltaWeightIH[0][j] ;
                dbg("WeightIH[%d][%d]:%lf = WeightIH[%d][%d]:%lf + DeltaWeightIH[%d][%d]:%lf\n",
                     0, j, WeightIH[0][j], 0, j, w, 0, j, DeltaWeightIH[0][j]);
#endif
                for( i = 1 ; i <= NumInput ; i++ ) { 
#if DBG == 0
                    DeltaWeightIH[i][j] = eta * Input[p][i] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
                    WeightIH[i][j] +=  DeltaWeightIH[i][j] ;
#else
                    momentum = alpha * DeltaWeightIH[i][j];
                    DeltaWeightIH[i][j] = (eta * Input[p][i] * DeltaH[j]) + momentum;
                    dbg("DeltaWeightIH[%d][%d]:%lf = (eta:%lf * Input[%d][%i]:%lf * DeltaH[%d]:%lf) + momentum:%lf\n",
                         i, j, DeltaWeightIH[i][j], eta, j, p, i, Input[p][i], DeltaH[j], momentum);
                    w = WeightIH[i][j];
                    WeightIH[i][j] =  WeightIH[i][j] + DeltaWeightIH[i][j] ;
                    dbg("WeightIH[%d][%d]:%lf = WeightIH[%d][%d]:%lf + DeltaWeightIH[%d][%d]:%lf\n",
                         i, j, WeightIH[i][j], i, j, w, i, j, DeltaWeightIH[i][j]);
#endif
                }
            }
            dbg("Update WeightsHO:\n");
            for( k = 1 ; k <= NumOutput ; k ++ ) {    /* update weights WeightHO */
                double momentum;
                double w;
#if DBG == 0
                DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k] ;
                WeightHO[0][k] += DeltaWeightHO[0][k] ;
#else
                momentum = alpha * DeltaWeightHO[0][k];
                DeltaWeightHO[0][k] = (eta * DeltaO[k]) + momentum;
                dbg("DeltaWeightHO[%d][%d]:%lf = (eta:%lf * DeltaO[%d]:%lf) + momentum:%lf BIAS\n",
                     0, k, DeltaWeightHO[0][k], eta, k, DeltaO[k], momentum);
                w = WeightHO[0][k];
                WeightHO[0][k] = WeightHO[0][k] + DeltaWeightHO[0][k] ;
                dbg("WeightHO[%d][%d]:%lf = WeightHO[%d][%d] + DeltaWeightHO[%d][%d]:%lf\n",
                     0, k, WeightHO[0][k], 0, k, w, 0, k, DeltaWeightHO[0][k]);
#endif
                for( j = 1 ; j <= NumHidden ; j++ ) {
#if DBG == 0
                    DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k] ;
                    WeightHO[j][k] += DeltaWeightHO[j][k] ;
#else
                    momentum = alpha * DeltaWeightHO[j][k];
                    DeltaWeightHO[j][k] = (eta * Hidden[p][j] * DeltaO[k]) + momentum ;
                    dbg("DeltaWeightHO[%d][%d]:%lf = (eta:%lf * Hidden[%d][%d]:%lf * DeltaO[%d]:%lf) + momentum:%lf\n",
                         j, k, DeltaWeightHO[j][k], eta, p, j, Hidden[p][j], k, DeltaO[k], momentum);
                    w = WeightHO[j][k];
                    WeightHO[j][k] = WeightHO[j][k] + DeltaWeightHO[j][k] ;
                    dbg("WeightHO[%d][%d]:%lf = WeightHO[%d][%d]:%lf +  DeltaWeightHO[%d][%d]:%lf\n",
                         j, k, WeightHO[j][k], j, k, w, j, k, DeltaWeightHO[j][k]);
#endif
                }
            }
        }
        if( epoch%100 == 0 ) fprintf(stdout, "\nEpoch %-5d :   Error = %f", epoch, Error) ;
        if( Error < 0.0004 ) break ;  /* stop learning when 'near enough' */
    }
    
    fprintf(stdout, "\n\nNETWORK DATA - EPOCH %d - Error %lf\n\nPat\t", epoch, Error) ;   /* print network outputs */
    for( i = 1 ; i <= NumInput ; i++ ) {
        fprintf(stdout, "Input%-4d\t", i) ;
    }
    for( k = 1 ; k <= NumOutput ; k++ ) {
        fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 1 ; p <= NumPattern ; p++ ) {        
    fprintf(stdout, "\n%d\t", p) ;
        for( i = 1 ; i <= NumInput ; i++ ) {
            fprintf(stdout, "%f\t", Input[p][i]) ;
        }
        for( k = 1 ; k <= NumOutput ; k++ ) {
            fprintf(stdout, "%f\t%f\t", Target[p][k], Output[p][k]) ;
        }
    }
    fprintf(stdout, "\n\nGoodbye!\n\n") ;
    return 0 ;
}

/*******************************************************************************/
