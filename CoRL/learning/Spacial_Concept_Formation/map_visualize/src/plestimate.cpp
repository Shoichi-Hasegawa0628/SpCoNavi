#include <iostream>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>
#include <vector>
#include <typeinfo>
#include <math.h>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "include/foperation.h"
#include "include/plestimate.h"
#include "include/mathmatic.h" 
#include <stdlib.h>
#include <stdio.h>
using namespace std;
using namespace cv;
int space_num;
//---Gaussian distribution
double ND_Gaussian(double* data,double *mu, double **sigma,const int dim)
{   
    double ans=0;
    double right=0.0,left=0.0;
    //vector_check(data,4);
    //---caluclate scalar
    double det;
    if(dim>2){
        det=ND_cofactor(sigma,dim);
    }else{det=TwoD_saras(sigma);
    }
    left=pow(2*M_PI,(dim/2));
    left*=sqrt(det);

    //--calculate inverse covariance matirx;
    double temp[dim][dim];
    for(int i=0;i<dim;i++){
       for(int j=0;j<dim;j++){
       temp[i][j]=sigma[i][j];
        } 
    }
    double *temp_sigma[dim];
    for(int i=0;i<dim;i++)temp_sigma[i]=temp[i];
    
    inv_matirx(temp_sigma,dim);
    
    //---matrix_chek(sigma,dim);
    double temp_data[dim];
    for(int i=0;i<dim;i++){
            temp_data[i]=data[i]-mu[i];
            
    }
    double value[dim]; 
    for(int i=0;i<dim;i++){
        value[i]=0.0;
        for(int j=0;j<dim;j++){
            value[i]+=temp_data[j]*temp_sigma[j][i];
        }
    
    right +=temp_data[i]*value[i];

    }
    right=-(right*0.5);
    ans=-log(left)+right;
    /*
    if(ans>0){
    //matrix_chek(sigma,4);
    cout<<"det: "<<det<<" ans:"<<endl;
    cout<<"left:"<<log(left)<<" right:"<<right<<" exp:"<<exp(ans)<<endl;
    //exit(-1);
    //.ans=-10000;
    }*/
    return ans;
}

int space_num_check(string dir){
        stringstream file;
        file <<dir<<"word/word_distribution"<<0<< ".txt";
        //cout<<file.str()<<endl;
        ifstream ifs(file.str().c_str());
        if(!ifs){
            cout<<file.str()<<" is not existed."<<endl;
            exit(-1);
        }
        string str;
        int line=0;
        while(getline(ifs,str)){
            string token;
            istringstream stream(str);
            
            while(getline(stream,token,',')){
                line++;
            }
        }
    return line;
}


void word_map_estimate(const Mat im,const Map m,char dir[],char dir_result[]){
    string d=dir;
    string result_dir=dir_result;
    string mu_dir=d+"/mu/";
    std::vector<string> mu_fileset=search_file(mu_dir,".csv");
    string colorfile="space_color.txt";
    int color[100][3];
    color_read(color,colorfile);

    const int class_num=mu_fileset.size();
    double mu_set[class_num][4],sigma[class_num][4][4],pi_set[class_num];
    mu_read(mu_dir,mu_set);
    //vector_check(mu_set[0],4);
    sigma_read(dir,sigma);

    pi_read(dir,pi_set);
    //vector_check(pi_set,class_num);
    space_num=space_num_check(dir);
    cout<<space_num<<endl;
    double *W[class_num];
    word_read(dir,W,space_num);
    mkdir(result_dir.c_str(),0777);
    //cout<<W[0][10]<<endl;
    double position[4];
    ///estimate
    double angle=0;
    cpy_array(position,m.coordinate);
    int result_num;
    cout<<"Every how many angles do you generate a map?"<<endl;
    cin>>result_num;
    cout<<"The number of "<<int(360/result_num)<<" maps are generated."<<endl;
    Place *p;
    //p[0] = new int[3];
   // _mkdir("result");
    p=new Place[class_num]; 
    
    for(int c=0;c<class_num;c++){
        double *sigma_[4];
        for(int i=0;i<4;i++){
            sigma_[i]=sigma[c][i];
            p[c].sigma[i]=sigma[c][i];
        }
        //matrix_chek(p[c].sigma,4);
        p[c].mu=mu_set[c];
        //vector_check(p[c].mu,4);
        p[c].W=W[c];

    }
    /*
    for(int j=0;j<2;j++){
    for(int i=0;i<3;i++){
        matrix_chek(p[i].sigma,4);
    }}*/
    //Generating color map
    struct timeval s, t;
    gettimeofday(&s, NULL);

    while(angle<360){
    int width=im.cols,high=im.rows;
    Mat cp_im= Mat(high, width, CV_8UC3);
    double data[4];
    double rad = angle * M_PI / 180.0;
    data[2]=sin(rad);
    data[3]=cos(rad);
    data[1]=position[1];
    while(high!=0){
        width=im.cols;
        data[0]=position[0];
        for(int i=0;i<width;i++){
            //white pixcel
            if(im.at<cv::Vec3b>(high-1,i)[0]==254){
                //cout<<position[0]<<endl;
                double prob_c[class_num],prob_word[class_num];
                zero_array(prob_c,class_num);
                zero_array(prob_word,space_num);

                for(int c=0;c<class_num;c++){
                    //cout<<"";
                    //cout<<"^------"<<c<<"------"<<endl;
                    prob_c[c]=ND_Gaussian(data,p[c].mu,p[c].sigma,4);
                }
                //vector_check(prob_c,class_num);
                    //matrix_chek(p[c].sigma,4);
                int max_c=argmax(prob_c,class_num);
                double max_p=prob_c[max_c];
                for(int c=0;c<class_num;c++){
                    prob_c[c]-=max_p;
                    prob_c[c]=exp(prob_c[c]);
                }

                double sum_prob=sum_array(prob_c,class_num);
                //cout<<sum_prob<<endl;
                for(int c=0;c<class_num;c++){
                    prob_c[c]=prob_c[c]/sum_prob;
                    prob_c[c] *=(pi_set[c]);
                    for(int w=0;w<space_num;w++){
                        prob_word[w]+=p[c].W[w]*prob_c[c];
                        
                    }
                    //vector_check(p[c].W,space_num);
                }
                //vector_check(prob_word,space_num);
                int max_class=2;
                max_class=argmax(prob_word,space_num);
                //max_class=9;
            
                //ND_Gaussian()
                cp_im.at<cv::Vec3b>(high-1,i)[0]=color[max_class][0];
                cp_im.at<cv::Vec3b>(high-1,i)[1]=color[max_class][1];
                cp_im.at<cv::Vec3b>(high-1,i)[2]=color[max_class][2];
            }
            //Other pixcel
            else{
                cp_im.at<cv::Vec3b>(high-1,i)=im.at<cv::Vec3b>(high-1,i);
            }
            data[0]+=0.05;
            //width--;
        }
        data[1]+=0.05;
        high--;
    }
    
    stringstream im_file;
    im_file <<result_dir<<"/color_word_map_"<<angle<<".jpg";
    imwrite(im_file.str(),cp_im);
    gettimeofday(&t, NULL);
    cout<<"saving "<<im_file.str()<<" was done."<<endl;
    double sec=(t.tv_sec - s.tv_sec) * 1000 + (t.tv_usec - s.tv_usec) / 1000;
    cout<<"time "<<sec/1000<<endl;
    angle=angle+result_num;
    //imshow("image",cp_im);
    
    }
 delete(*W);
 delete(p);
}

void word_map_estimate_one_angle(const Mat im,const Map m,char dir[],char dir_result[]){
    string d=dir;
    string result_dir=dir_result;
    string mu_dir=d+"/mu/";
    std::vector<string> mu_fileset=search_file(mu_dir,".csv");
    string colorfile="space_color.txt";
    int color[100][3];
    color_read(color,colorfile);

    const int class_num=mu_fileset.size();
    double mu_set[class_num][4],sigma[class_num][4][4],pi_set[class_num];
    mu_read(mu_dir,mu_set);
    //vector_check(mu_set[0],4);
    sigma_read(dir,sigma);

    pi_read(dir,pi_set);
    //vector_check(pi_set,class_num);
    space_num=space_num_check(dir);
    cout<<space_num<<endl;
    double *W[class_num];
    word_read(dir,W,space_num);
    mkdir(result_dir.c_str(),0777);
    //cout<<W[0][10]<<endl;
    double position[4];
    ///estimate
    
    cpy_array(position,m.coordinate);
    int result_num;
    Place *p;
    //p[0] = new int[3];
   // _mkdir("result");
    p=new Place[class_num]; 
    
    for(int c=0;c<class_num;c++){
        double *sigma_[4];
        for(int i=0;i<4;i++){
            sigma_[i]=sigma[c][i];
            p[c].sigma[i]=sigma[c][i];
        }
        //matrix_chek(p[c].sigma,4);
        p[c].mu=mu_set[c];
        //vector_check(p[c].mu,4);
        p[c].W=W[c];

    }
    /*
    for(int j=0;j<2;j++){
    for(int i=0;i<3;i++){
        matrix_chek(p[i].sigma,4);
    }}*/
    //Generating color map
    struct timeval s, t;
    gettimeofday(&s, NULL);

    
    int width=im.cols,high=im.rows;
    Mat cp_im= Mat(high, width, CV_8UC3);
    double data[4];
    

    data[1]=position[1];
    while(high!=0){
        width=im.cols;
        data[0]=position[0];
        for(int i=0;i<width;i++){
            //white pixcel
            if(im.at<cv::Vec3b>(high-1,i)[0]==254){
                //cout<<position[0]<<endl;
                double prob_c[class_num],prob_word[class_num];
                zero_array(prob_c,class_num);
                zero_array(prob_word,space_num);
                
                for(int angle=0;angle<360;angle++){
                double rad = angle * M_PI / 180.0;
                data[2]=sin(rad);
                data[3]=cos(rad);
                for(int c=0;c<class_num;c++){
                    //cout<<"";
                    //cout<<"^------"<<c<<"------"<<endl;
                    prob_c[c]=ND_Gaussian(data,p[c].mu,p[c].sigma,4);
                }
                //vector_check(prob_c,class_num);
                    //matrix_chek(p[c].sigma,4);
                int max_c=argmax(prob_c,class_num);
                double max_p=prob_c[max_c];
                for(int c=0;c<class_num;c++){
                    prob_c[c]-=max_p;
                    prob_c[c]=exp(prob_c[c]);
                    }

                double sum_prob=sum_array(prob_c,class_num);
                //cout<<sum_prob<<endl;
                for(int c=0;c<class_num;c++){
                    prob_c[c]=prob_c[c]/sum_prob;
                    prob_c[c] *=(pi_set[c]);
                    for(int w=0;w<space_num;w++){
                        prob_word[w]+=p[c].W[w]*prob_c[c];
                        }
                    //vector_check(p[c].W,space_num);
                    }
                }
                //vector_check(prob_word,space_num);
                int max_class;
                max_class=argmax(prob_word,space_num);
                //max_class=9;
                cout<<i<<" "<<high<<endl;
                //ND_Gaussian()
                cp_im.at<cv::Vec3b>(high-1,i)[0]=color[max_class][0];
                cp_im.at<cv::Vec3b>(high-1,i)[1]=color[max_class][1];
                cp_im.at<cv::Vec3b>(high-1,i)[2]=color[max_class][2];
            }
            //Other pixcel
            else{
                cp_im.at<cv::Vec3b>(high-1,i)=im.at<cv::Vec3b>(high-1,i);
            }
            data[0]+=0.05;
            //width--;
        }
        data[1]+=0.05;
        high--;
    }
    
    stringstream im_file;
    im_file <<result_dir<<"/color_word_map_all_direction.jpg";
    imwrite(im_file.str(),cp_im);
    gettimeofday(&t, NULL);
    cout<<"saving "<<im_file.str()<<" was done."<<endl;
    double sec=(t.tv_sec - s.tv_sec) * 1000 + (t.tv_usec - s.tv_usec) / 1000;
    cout<<"time "<<sec/1000<<endl;
    
    //imshow("image",cp_im);
 delete(*W);
 delete(p);
}

void word_map_estimate_no_angle(const Mat im,const Map m,char dir[],char dir_result[]){
    string d=dir;
    string result_dir=dir_result;
    string mu_dir=d+"/mu/";
    std::vector<string> mu_fileset=search_file(mu_dir,".csv");
    string colorfile="space_color.txt";
    int color[100][3];
    color_read(color,colorfile);

    const int class_num=mu_fileset.size();
    double mu_set[class_num][2],sigma[class_num][2][2],pi_set[class_num];
    two_mu_read(mu_dir,mu_set);
    //vector_check(mu_set[0],4);
    two_sigma_read(dir,sigma);

    pi_read(dir,pi_set);
    //vector_check(pi_set,class_num);
    space_num=space_num_check(dir);
    cout<<space_num<<endl;
    double *W[class_num];
    word_read(dir,W,space_num);
    mkdir(result_dir.c_str(),0777);
    //cout<<W[0][10]<<endl;
    double position[2];
    ///estimate
    
    cpy_array(position,m.coordinate);
    int result_num;
    Place_Two *p;
    //p[0] = new int[3];
   // _mkdir("result");
    p=new Place_Two[class_num]; 
    
    for(int c=0;c<class_num;c++){
        double *sigma_[2];
        for(int i=0;i<2;i++){
            sigma_[i]=sigma[c][i];
            p[c].sigma[i]=sigma[c][i];
        }
        //matrix_chek(p[c].sigma,4);
        p[c].mu=mu_set[c];
        //vector_check(p[c].mu,4);
        p[c].W=W[c];

    }
    /*
    for(int j=0;j<2;j++){
    for(int i=0;i<3;i++){
        matrix_chek(p[i].sigma,4);
    }}*/
    //Generating color map
    struct timeval s, t;
    gettimeofday(&s, NULL);

    
    int width=im.cols,high=im.rows;
    Mat cp_im= Mat(high, width, CV_8UC3);
    double data[2];
    

    data[1]=position[1];
    while(high!=0){
        width=im.cols;
        data[0]=position[0];
        for(int i=0;i<width;i++){
            //white pixcel
            if(im.at<cv::Vec3b>(high-1,i)[0]==254){
                //cout<<position[0]<<endl;
                double prob_c[class_num],prob_word[class_num];
                zero_array(prob_c,class_num);
                zero_array(prob_word,space_num);
                

                for(int c=0;c<class_num;c++){
                    //cout<<"";
                    //cout<<"^------"<<c<<"------"<<endl;
                    prob_c[c]=ND_Gaussian(data,p[c].mu,p[c].sigma,2);
                }
                //vector_check(prob_c,class_num);
                    //matrix_chek(p[c].sigma,4);
                int max_c=argmax(prob_c,class_num);
                double max_p=prob_c[max_c];
                for(int c=0;c<class_num;c++){
                    prob_c[c]-=max_p;
                    prob_c[c]=exp(prob_c[c]);
                    }

                double sum_prob=sum_array(prob_c,class_num);
                //cout<<sum_prob<<endl;
                
                for(int c=0;c<class_num;c++){
                    prob_c[c]=prob_c[c]/sum_prob;
                    prob_c[c] *=(pi_set[c]);
                    for(int w=0;w<space_num;w++){
                        prob_word[w]+=p[c].W[w]*prob_c[c];
                        
                        }
                    //vector_check(p[c].W,space_num);
                    }
                

                
                int max_class;
                max_class=argmax(prob_word,space_num);

                double word_sum=0;
                for(int w=0;w<space_num;w++){
                    word_sum=word_sum+prob_word[w];
                }
                //max_class=9;
                //cout<<prob_word[max_class]<<endl;
                cout<<i<<" "<<high<<endl;
                //cout<<sum_word<<endl;
                //ND_Gaussian()
                prob_word[max_class]=prob_word[max_class]/word_sum;
                cout<<prob_word[max_class]<<endl;
                //if(prob_word[max_class]>0.9){
                cp_im.at<cv::Vec3b>(high-1,i)[0]=color[max_class][0];
                cp_im.at<cv::Vec3b>(high-1,i)[1]=color[max_class][1];
                cp_im.at<cv::Vec3b>(high-1,i)[2]=color[max_class][2];
                //}else{
                //cp_im.at<cv::Vec3b>(high-1,i)[0]=254;
                //cp_im.at<cv::Vec3b>(high-1,i)[1]=254;
                //cp_im.at<cv::Vec3b>(high-1,i)[2]=254;

                //}
            }
            //Other pixcel
            else{
                cp_im.at<cv::Vec3b>(high-1,i)=im.at<cv::Vec3b>(high-1,i);
            }
            data[0]+=0.05;
            //width--;
        }
        data[1]+=0.05;
        high--;
    }
    
    stringstream im_file;
    im_file <<result_dir<<"/color_word_map_no_angle.jpg";
    imwrite(im_file.str(),cp_im);
    gettimeofday(&t, NULL);
    cout<<"saving "<<im_file.str()<<" was done."<<endl;
    double sec=(t.tv_sec - s.tv_sec) * 1000 + (t.tv_usec - s.tv_usec) / 1000;
    cout<<"time "<<sec/1000<<endl;
    
    //imshow("image",cp_im);
 delete(*W);
 delete(p);
}



