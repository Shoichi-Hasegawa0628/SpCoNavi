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
#include "include/position_estimate.h"
#include <stdlib.h>
#include <stdio.h>
using namespace std;
using namespace cv;
const int region_num=80,concept_num=150;
const int place_num=53;
void trans_pi_read(std::string dir,double pi_set[concept_num][region_num]){
    stringstream file;
    file <<dir<<"pi.csv";
    ifstream ifs(file.str().c_str());
    if(!ifs){
            cout<<dir<<file.str()<< " is not existed."<<endl;
            exit(-1);
        }
    
    string str;

    int j=0;
    while(getline(ifs,str)){
        string line;
        istringstream stream(str);    
        while(getline(stream,line,',')){
            string line_str=line;
            std::vector<std::string> value;
            value=split(line,' ');
            for(int d=0;d<value.size();d++){
                double temp=atof(value[d].c_str());
                pi_set[j][d]=temp;
                //cout<<" "<<pi_set[j][d];
                }

        //cout<<endl;
        }
        j++;
     }
     //cout<<"]"<<endl;
}

void trans_phi_n_read(std::string dir,double phi_w_set[concept_num][place_num]){
    stringstream file;
    file <<dir<<"../phi_n.csv";
    ifstream ifs(file.str().c_str());
    if(!ifs){
            cout<<dir<<file.str()<< " is not existed."<<endl;
            exit(-1);
        }
    
    string str;

    int j=0;
    while(getline(ifs,str)){
        string line;
        istringstream stream(str);    
        while(getline(stream,line,',')){
            string line_str=line;
            std::vector<std::string> value;
            value=split(line,' ');
            //cout<<"class"<<j<<endl;
            for(int d=0;d<value.size();d++){
                double temp=atof(value[d].c_str());
                phi_w_set[j][d]=temp;
                //cout<<" "<<phi_w_set[j][d];
                }

        //cout<<endl;
        }
        j++;
     }
     //cout<<"]"<<endl;
}


void position_estimate_trans(const Mat im,const Map m,char dir[],char dir_result[]){
    string d=dir;
    string result_dir=dir_result;
    string mu_dir=d+"/mu/";

    std::vector<string> mu_fileset=search_file(mu_dir,".csv");
    cout<<1<<endl;
    int name_dim=53;
    
    const int class_num=mu_fileset.size();
    cout<<"region_num:"<<class_num<<endl;
    double mu_set[class_num][4],sigma[class_num][4][4];
    double pi_set[concept_num][region_num];
    mu_read(mu_dir,mu_set);
    //vector_check(mu_set[0],4);

    sigma_read(dir,sigma);
    trans_pi_read(dir,pi_set);
    //vector_check(pi_set,class_num);
    int i;
    double phi_n[concept_num][place_num]; 
    trans_phi_n_read(dir,phi_n);

    mkdir(result_dir.c_str(),0777);
    
    double position[4];
    
    cpy_array(position,m.coordinate);
    int result_num;
    cout<<"Every how many angles do you generate a map?"<<endl;
    cin>>result_num;
    cout<<"The number of "<<int(360/result_num)<<" maps are generated."<<endl;
    Place *p;
    p=new Place[class_num]; 

    for(int c=0;c<region_num;c++){
        double *sigma_[4];
        for(int i=0;i<4;i++){
            sigma_[i]=sigma[c][i];
            p[c].sigma[i]=sigma[c][i];
        }
        p[c].mu=mu_set[c];
    }
    
    //Generating color map
    struct timeval s, t;
    gettimeofday(&s, NULL);
    //int w=1;
    for(int w=0;w<place_num;w++){
    stringstream save_dir;
    save_dir <<result_dir<<"/"<<w<<"/";
    mkdir(save_dir.str().c_str(),0777);
    double angle=0;

    double prob_all=0;
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
                //double prob_c[class_num],prob_word[class_num];
                double prob=0,r_to_x[region_num];

                
                for(int r=0;r<region_num;r++){
                    r_to_x[r]=ND_Gaussian(data,p[r].mu,p[r].sigma,4);
                }

                int max_r=argmax(r_to_x,region_num);
                double max_p=r_to_x[max_r];
                for(int r=0;r<region_num;r++){
                    r_to_x[r]-=max_p;
                    r_to_x[r]=exp(r_to_x[r]);
                }

                double sum_prob=sum_array(r_to_x,region_num);
                for(int r=0;r<region_num;r++){
                    r_to_x[r]=r_to_x[r]/sum_prob;
                    //cout<<"r_t:"<<r_to_x[r]<<endl;
                    double temp=0;
                    for(int c=0;c<concept_num;c++){
                        //cout<<c<<endl;
                        temp=temp+(r_to_x[r]*phi_n[c][w]*pi_set[c][r]);   
                    }

                    prob=prob+temp;
                    //cout<<r<<endl;
                }   
                prob_all=prob_all+prob;
                prob=prob*10;
                
                //cout<<prob<<endl;
                if(prob>0.4){
                cp_im.at<cv::Vec3b>(high-1,i)[0]=255*(1-prob);//255*(1-prob);
                cp_im.at<cv::Vec3b>(high-1,i)[1]=0;//255*(1-prob);
                cp_im.at<cv::Vec3b>(high-1,i)[2]=(255*prob);//;
                }else{
                cp_im.at<cv::Vec3b>(high-1,i)[0]=255;//255*(1-prob);
                cp_im.at<cv::Vec3b>(high-1,i)[1]=255;//255*(1-prob);
                cp_im.at<cv::Vec3b>(high-1,i)[2]=255;//;

                }
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
    im_file <<save_dir.str().c_str()<<"/position_estimate_"<<angle<<".jpg";
    imwrite(im_file.str(),cp_im);
    gettimeofday(&t, NULL);
    cout<<"saving "<<im_file.str()<<" was done."<<endl;
    double sec=(t.tv_sec - s.tv_sec) * 1000 + (t.tv_usec - s.tv_usec) / 1000;
    cout<<"time "<<sec/1000<<endl;
    angle=angle+result_num;
    //imshow("image",cp_im);
    
    }

 cout<<prob_all<<endl;
}
 delete(p);
 
}
void position_estimate_trans_new(const Mat im,const Map m,char dir[],char dir_result[]){
    string d=dir;
    string result_dir=dir_result;
    string mu_dir=d+"/mu/";

    std::vector<string> mu_fileset=search_file(mu_dir,".csv");
    cout<<1<<endl;
    int name_dim=53;
    
    const int class_num=mu_fileset.size();
    cout<<"region_num:"<<class_num<<endl;
    double mu_set[class_num][4],sigma[class_num][4][4];
    double pi_set[concept_num][region_num];
    mu_read(mu_dir,mu_set);
    //vector_check(mu_set[0],4);

    sigma_read(dir,sigma);
    trans_pi_read(dir,pi_set);
    //vector_check(pi_set,class_num);
    int i;
    double phi_n[concept_num][place_num]; 
    trans_phi_n_read(dir,phi_n);

    mkdir(result_dir.c_str(),0777);
    
    double position[4];
    
    cpy_array(position,m.coordinate);
    int result_num;
    cout<<"Every how many angles do you generate a map?"<<endl;
    cin>>result_num;
    cout<<"The number of "<<int(360/result_num)<<" maps are generated."<<endl;
    Place *p;
    p=new Place[class_num]; 

    for(int c=0;c<region_num;c++){
        double *sigma_[4];
        for(int i=0;i<4;i++){
            sigma_[i]=sigma[c][i];
            p[c].sigma[i]=sigma[c][i];
        }
        p[c].mu=mu_set[c];
    }
    
    //Generating color map
    struct timeval s, t;
    gettimeofday(&s, NULL);
    //int w=1;
    for(int w=0;w<place_num;w++){
    stringstream save_dir;
    save_dir <<result_dir<<"/"<<w<<"/";
    mkdir(save_dir.str().c_str(),0777);
    double angle=0;

    double prob_all=0;
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
                //double prob_c[class_num],prob_word[class_num];
                double prob=0,r_to_x[region_num];

                
                for(int r=0;r<region_num;r++){
                    r_to_x[r]=ND_Gaussian(data,p[r].mu,p[r].sigma,4);
                }

                int max_r=argmax(r_to_x,region_num);
                double max_p=r_to_x[max_r];
                for(int r=0;r<region_num;r++){
                    r_to_x[r]-=max_p;
                    r_to_x[r]=exp(r_to_x[r]);
                }

                double sum_prob=sum_array(r_to_x,region_num);
                for(int r=0;r<region_num;r++){
                    r_to_x[r]=r_to_x[r]/sum_prob;
                    //cout<<"r_t:"<<r_to_x[r]<<endl;
                    double temp=0;
                    for(int c=0;c<concept_num;c++){
                        //cout<<c<<endl;
                        temp=temp+(r_to_x[r]*phi_n[c][w]*pi_set[c][r]);   
                    }

                    prob=prob+temp;
                    //cout<<r<<endl;
                }   
                prob_all=prob_all+prob;
                prob=prob*10;
                
                //cout<<prob<<endl;
                if(prob>0.4){
                cp_im.at<cv::Vec3b>(high-1,i)[0]=255*(1-prob);//255*(1-prob);
                cp_im.at<cv::Vec3b>(high-1,i)[1]=0;//255*(1-prob);
                cp_im.at<cv::Vec3b>(high-1,i)[2]=(255*prob);//;
                }else{
                cp_im.at<cv::Vec3b>(high-1,i)[0]=255;//255*(1-prob);
                cp_im.at<cv::Vec3b>(high-1,i)[1]=255;//255*(1-prob);
                cp_im.at<cv::Vec3b>(high-1,i)[2]=255;//;

                }
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
    im_file <<save_dir.str().c_str()<<"/position_estimate_"<<angle<<".jpg";
    imwrite(im_file.str(),cp_im);
    gettimeofday(&t, NULL);
    cout<<"saving "<<im_file.str()<<" was done."<<endl;
    double sec=(t.tv_sec - s.tv_sec) * 1000 + (t.tv_usec - s.tv_usec) / 1000;
    cout<<"time "<<sec/1000<<endl;
    angle=angle+result_num;
    //imshow("image",cp_im);
    
    }

 cout<<prob_all<<endl;
}
 delete(p);
 
}