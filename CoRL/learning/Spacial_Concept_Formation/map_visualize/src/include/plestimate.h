#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
typedef struct Place{
    int a;
    double *mu;
    double *sigma[4];
    double *fi;
    double *W;
}Place;

typedef struct Place_Two{
    int a;
    double *mu;
    double *sigma[4];
    double *fi;
    double *W;
}Place_Two;
double ND_Gaussian(double *data,double *mu, double **sigma,const int dim);

int space_num_check(string dir);

void word_map_estimate(const Mat im,const Map m,char dir[],char dir_result[]);

void word_map_estimate_one_angle(const Mat im,const Map m,char dir[],char dir_result[]);

void word_map_estimate_no_angle(const Mat im,const Map m,char dir[],char dir_result[]);
