#include <stdio.h>
#include <string.h>
#include <synchapi.h>
#include <math.h>
#include "mpi.h"

#define TEST_MPI_FUNC( FuNAME, ITER_NUMBER, RANK, ROOT_RANK, ... )          \
    {                                                                       \
        if (RANK == ROOT_RANK) {                                            \
            double time_arr[128] = {};                                      \
            double time_start = 0;                                          \
            double time_end = 0;                                            \
            double time_total = 0;                                          \
            double time_med = 0;                                            \
            double time_STD = 0;                                            \
            double time_dev = 0;                                            \
                                                                            \
            for (int i = 0; i < ITER_NUMBER; i++)                           \
            {                                                               \
                MPI_Barrier (MPI_COMM_WORLD);                               \
                time_start = MPI_Wtime();                                   \
                MPI_##FuNAME ( __VA_ARGS__ );                               \
                time_end = MPI_Wtime();                                     \
                                                                            \
                time_arr[i] = time_end - time_start;                        \
                time_total += time_arr[i];                                  \
                                                                            \
                MPI_Barrier (MPI_COMM_WORLD);                               \
            }                                                               \
                                                                            \
            time_med = time_total / ITER_NUMBER;                            \
            for (int i = 0; i < ITER_NUMBER; i++)                           \
            {                                                               \
                time_dev = time_med - time_arr[i];                          \
                time_STD += time_dev * time_dev;                            \
            }                                                               \
            time_STD = sqrt (time_STD / (ITER_NUMBER - 1));                 \
                                                                            \
            printf ("%s executing time: %lf  |  STD: %lf  |  Wtick: %lf\n", \
                    #FuNAME,                                                \
                    time_total / ITER_NUMBER,                               \
                    time_STD,                                               \
                    MPI_Wtick());                                           \
                                                                            \
            MPI_Barrier (MPI_COMM_WORLD);                                   \
        }                                                                   \
        else {                                                              \
            for (int i = 0; i < ITER_NUMBER; i++)                           \
            {                                                               \
                MPI_Barrier (MPI_COMM_WORLD);                               \
                MPI_##FuNAME ( __VA_ARGS__ );                               \
                MPI_Barrier (MPI_COMM_WORLD);                               \
            }                                                               \
            MPI_Barrier (MPI_COMM_WORLD);                                   \
        }                                                                   \
    }

// Run with MPIEXEC -np <n> <program_name> <iter_number> <root_rank>
int main (int argc, char** argv)
{
    char* endptr = 0;
    int iterations_number = strtol (argv[1], &endptr, 10);
    int root_rank = strtol (argv[2], &endptr, 10);

    int rank = 0, size = 0;
    int rank_buffer = 0;

    int sum = 0;
    int int_buffer = 0;
    int arr_buffer[128] = {};


    MPI_Init (&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    rank_buffer = rank;

    TEST_MPI_FUNC (Bcast, iterations_number, rank, root_rank,
                   &rank_buffer, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    TEST_MPI_FUNC (Reduce, iterations_number, rank, root_rank,
                   &rank, &sum, 1, MPI_INT, MPI_SUM, root_rank, MPI_COMM_WORLD);
    TEST_MPI_FUNC (Gather, iterations_number, rank, root_rank,
                   &rank, 1, MPI_INT, &arr_buffer, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    TEST_MPI_FUNC (Scatter, iterations_number, rank, root_rank,
                   &rank, 1, MPI_INT, &int_buffer, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
