// RUN: %ldc -c -O4 -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// REQUIRES: Polly

import ldc.attributes;
import ldc.intrinsics;


enum N = 3200;
enum M = 2600;

@polly void correlate(float n,ref float[N][M] data, out float[M][M] corr)
{
    float[M] mean = void, stddev = void;
    foreach(j;0 .. M)
    {
        mean[j] = 0.0;
        foreach(i; 0 .. N)
            mean[j] += data[i][j];
        mean[j] /= n;
    }

    foreach(j;0 .. M)
    {
        stddev[j] = 0.0;
        foreach(i; 0 .. N)
            stddev[j] += (data[i][j]-mean[j])*(data[i][j]-mean[j]);
        stddev[j] /= n;
        stddev[j] = llvm_sqrt(stddev[j]);
        stddev[j] = max(0.1,stddev[j]);
    }

    foreach(i; 0 .. N)
        foreach(j;0 .. M)
        {
            data[i][j] -= mean[j];
            data[i][j] /= llvm_sqrt(n)* stddev[j];
        }
    foreach(i; 0 .. M-1)
    {
        corr[i][i] = 1.0;
        for (auto j = i+1; j < M; j++)
        {
            corr[i][j] = 0.0;
            for (auto k = 0; k < N; k++)
            corr[i][j] += data[k][i] * data[k][j];
            corr[j][i] = corr[i][j];
        }
    }
    corr[M-1][M-1] = 1.0;
}
