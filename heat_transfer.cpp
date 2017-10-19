#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <omp.h>

// Load the multidimensional array template
#include "arrayff.hxx"

// Load the template functions to draw the hot and cold spots
#include "draw.hxx"


/*
 *                                SPECIFICATION
 *  The programe aims to simulate heat transfermation via OpenMP.
 *
 *  Usage: heat_transfer [npix] [num_threads]
 *  npix is the width and height of output plate, if not specified, it is 100.
 *  num_threads is the threads that will be created while computing, it should be in range [1, max_threads], by default, it is max_threads.
 */


using namespace std;

int main(int argc, char* argv[])
{
  const float tol = 0.00001;
  int max_threads = omp_get_max_threads();
  int npix = 100; // Be defualt the width and height of output plate is 100
  int num_threads = max_threads; // Be defualt we use all threads that we can use
  if(argc >= 2)
    npix = atoi(argv[1]);
  if(argc >= 3)
    num_threads = atoi(argv[2]);
  const int npixx = npix;
  const int npixy = npix;
  double t1 = omp_get_wtime();

  if(num_threads < 1)  // At least we should have 1 thread
    num_threads = 1;
  if(max_threads < num_threads) // Threads number could not exceed max_threads
    num_threads = max_threads;

  omp_set_num_threads(num_threads);

  int *covered = new int[num_threads];
  for(int i = 0; i < num_threads; i++) covered[i] = 0;
  Array<float, 2> h(npixy, npixy), g(npixy, npixx);

  const int nrequired = npixx * npixy;
  const int ITMAX = 100000;

  int nconverged = 0;
  fix_boundaries2(h);
  dump_array<float, 2>(h, "plate0.fit");

  // Divide npixy into num_threads parts, each part contains 'step' values/
  int step = npixy / num_threads;
  int total_iters = 0;

  #pragma omp parallel
  {
      int thread_num = omp_get_thread_num();
      int begin1 = thread_num * step;   // The begin index for updating g
      int end1 = (thread_num + 1) * step; // The end index for updating g

      if(thread_num == 0)   // The border of the image will not be considered when update g.
          begin1 = 1;
      if(thread_num == num_threads -1)
          end1 = npixy -1;

      int begin2 = begin1;    // The begin index for updating h
      int end2 = end1;        // The end index for updating h
      if(thread_num == 0)
          begin2 = 0;
      if(thread_num == num_threads -1)
          end2 = npixy;

      for(int iter = 0; iter < ITMAX ; iter ++)
      {
          if(nconverged >= nrequired)
         {
              if(total_iters == 0)
                  total_iters = iter;  // Save the iteration count
              break;
          }
          for(int y = begin1; y < end1; y ++)
          {
              for (int x = 1; x < npixx-1; ++x) {
                g(y, x) = 0.25 * (h(y, x-1) + h(y, x+1) + h(y-1, x) + h(y+1,x));
              }
          }
          nconverged =0;
          #pragma omp barrier // Make sure all g values has been updated

          fix_boundaries2(g);
          int covered_by_thread = 0;
          for (int y = begin2; y < end2; ++y) {
            for (int x = 0; x < npixx; ++x) {
              float dhg = std::fabs(g(y, x) - h(y, x));
              if (dhg < tol)
                  covered_by_thread ++;
              h(y, x) = g(y, x);
            }
          }
          nconverged += covered_by_thread;
          #pragma omp barrier // Make sure nconverged has been updated by all threads
      }
  }

  dump_array<float, 2>(h, "plate1.fit");
  std::cout << "Required " << total_iters << " iterations" << endl;
  double t2 = omp_get_wtime();
  float time = (t2 - t1);// / (CLOCKS_PER_SEC * 1.0);
  std::cout<<"time used: "<< time << " seconds, threads used:" << num_threads << endl;
}
