#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "include/foperation.h"
#include "include/plestimate.h"
#include "include/mathmatic.h" 
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{   
  Map m;
  m=read_parameter(argv[1]);
  Mat im=cv::imread(m.name);
  if (im.data == 0){
    cout<<"Map image file is not existed."<<endl;
    exit(-1);
  }
  word_map_estimate_no_angle(im,m,argv[1],argv[2]);
  //word_map_estimate_one_angle(im,m,argv[1],argv[2]);
return 0;
}
