#include "filter.hpp"

int main(int argc, char **argv)
{
    int itemsPerCore;
    int rank, size, tag, rc, i;
    double startWtime, endWtime;
    MPI_Status status;

    rc = MPI_Init(&argc, &argv);
    rc = MPI_Comm_size(MPI_COMM_WORLD, &size);
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    tag = 100;


    std::shared_ptr<cv::Mat> image;
    std::shared_ptr<cv::Mat> result;
    std::shared_ptr< std::vector<uchar> > subImage;
    std::shared_ptr< std::vector<uchar> > subResult;

    if (rank == 0) {
        image.reset(new cv::Mat(cv::imread(argv[1], cv::IMREAD_GRAYSCALE)));
        result.reset(new cv::Mat(image->rows, image->cols, CV_8UC1));

        itemsPerCore = image->total() / size;

        for (int i = 1; i < size; i++ )
            rc = MPI_Send(&itemsPerCore, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
    }
    else
        rc = MPI_Recv(&itemsPerCore, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);


    subImage.reset(new std::vector<uchar>(itemsPerCore));
    subResult.reset(new std::vector<uchar>(itemsPerCore));

    startWtime = MPI_Wtime();

    if ( rank == 0)
        MPI_Scatter(image->data, itemsPerCore, MPI_BYTE, subImage->data(), itemsPerCore, MPI_BYTE, 0, MPI_COMM_WORLD);
    else 
        MPI_Scatter(NULL, 0, NULL, subImage->data(), itemsPerCore, MPI_BYTE, 0, MPI_COMM_WORLD);

    std::shared_ptr<cv::Mat> aux_image(new cv::Mat(*subImage));
    std::shared_ptr<cv::Mat> aux_result(new cv::Mat(*subResult));
    applyFilter(aux_image, aux_result);

    MPI_Barrier(MPI_COMM_WORLD);

    if ( rank == 0)
        MPI_Gather(aux_result->data, itemsPerCore, MPI_BYTE, result->data, itemsPerCore, MPI_BYTE, 0, MPI_COMM_WORLD);
    else 
        MPI_Gather(aux_result->data, itemsPerCore, MPI_BYTE, NULL, 0, NULL, 0, MPI_COMM_WORLD);

    endWtime = MPI_Wtime();


    if (rank == 0) {
        cv::imwrite("result.png", *result);
        std::cout << "Elapsed time: " << endWtime - startWtime << "s" << std::endl;
    }

    rc = MPI_Finalize();
    return 0;
}