#include <iostream>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <typeinfo>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include "include/foperation.h"
#include "include/plestimate.h"
#include "include/mathmatic.h" 
using namespace std;

//***********************Search file function********************************
std::vector<std::string> search_file(std::string dir,std::string extension)
{
    DIR* dp=opendir(dir.c_str());//ディリクトリの取得 
    dirent* entry;
    std::vector<std::string> file_set;
    if (dp!=NULL)
    {
     while(entry!=NULL)
        {
            entry = readdir(dp);
            if(entry==NULL)
             {break;}
            else if(strstr(entry->d_name,extension.c_str())!=NULL)
             {       
                //cout <<dir<< entry->d_name << std::endl;
                std::string file=dir+entry->d_name;
                file_set.push_back(file);
             }
        }
    }else
    {cout<<dir<<" is not existed."<<endl;
        exit(1);
    }
 return file_set;
}

//*******************Spliting function*********************************
std::vector<std::string>split(const std::string &str,char sep){
    std::vector<std::string> v;
    std::stringstream ss(str);
    std::string buffer;
    while( std::getline(ss, buffer, sep) ) {
        v.push_back(buffer);
    }
    return v;
}
//***********************Reading  function********************************
void  mu_read(std::string mu_dir,double mu_set[][4]){
    //string mu_dir=dir+"/mu/";
    cout<<""<<endl;
    const int dim=4;
    std::vector<string> mu_fileset=search_file(mu_dir,".csv");
    int file_num=mu_fileset.size();

    for(int i=0;i<mu_fileset.size();i++){
        stringstream file;
        file << mu_dir <<"gauss_mu"<<i<< ".csv";
        
        ifstream ifs(file.str().c_str());
        if(!ifs){
            cout<<file.str()<<" is not existed."<<endl;
            exit(-1);
        }
        string str;
        int j=0;
        while(getline(ifs,str)){
            string token;
            istringstream stream(str);
            //cout<<str<<endl;
            while(getline(stream,token,',')){
                double temp=atof(token.c_str());
                mu_set[i][j]=temp;
                //cout<<i<<" "<<j<<" "<<mu_set[i][j];
                
            }
            j++;
        }
        //cout<<endl;
    }

}

void  two_mu_read(std::string mu_dir,double mu_set[][2]){
    //string mu_dir=dir+"/mu/";
    cout<<""<<endl;
    const int dim=2;
    std::vector<string> mu_fileset=search_file(mu_dir,".csv");
    int file_num=mu_fileset.size();

    for(int i=0;i<mu_fileset.size();i++){
        stringstream file;
        file << mu_dir <<"gauss_mu"<<i<< ".csv";
        
        ifstream ifs(file.str().c_str());
        if(!ifs){
            cout<<file.str()<<" is not existed."<<endl;
            exit(-1);
        }
        string str;
        int j=0;
        while(getline(ifs,str)){
            string token;
            istringstream stream(str);
            //cout<<str<<endl;
            while(getline(stream,token,',')){
                double temp=atof(token.c_str());
                mu_set[i][j]=temp;
                //cout<<i<<" "<<j<<" "<<mu_set[i][j];
                
            }
            j++;
        }
        //cout<<endl;
    }

}

void  sigma_read(std::string dir,double sigma_set[][4][4]){
    string sigma_dir=dir+"/sigma/";
    const int dim=4;
    
    std::vector<string> mu_fileset=search_file(sigma_dir,".csv");
    int file_num=mu_fileset.size();
    //cout<<file_num<<endl;
    for(int i=0;i<mu_fileset.size();i++){
        stringstream file;
        file << sigma_dir <<"gauss_sigma"<<i<< ".csv";
        ifstream ifs(file.str().c_str());
        if(!ifs){
            cout<<sigma_dir<<file.str()<< " is not existed."<<endl;
            exit(-1);
        }
        string str;
        //cout<<i<<endl;
        //cout<<"[";
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
                    sigma_set[i][j][d]=temp;
                    //cout<<" "<<sigma_set[i][j][d];
                    }

        //cout<<endl;
        }
        j++;
     }
     //cout<<"]"<<endl;
   }
}
void  two_sigma_read(std::string dir,double sigma_set[][2][2]){
    string sigma_dir=dir+"/sigma/";
    const int dim=2;
    
    std::vector<string> mu_fileset=search_file(sigma_dir,".csv");
    int file_num=mu_fileset.size();
    //cout<<file_num<<endl;
    for(int i=0;i<mu_fileset.size();i++){
        stringstream file;
        file << sigma_dir <<"gauss_sgima"<<i<< ".csv";
        ifstream ifs(file.str().c_str());
        if(!ifs){
            cout<<sigma_dir<<file.str()<< " is not existed."<<endl;
            exit(-1);
        }
        string str;
        //cout<<i<<endl;
        //cout<<"[";
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
                    sigma_set[i][j][d]=temp;
                    //cout<<" "<<sigma_set[i][j][d];
                    }

        //cout<<endl;
        }
        j++;
     }
     //cout<<"]"<<endl;
   }
}

void word_read(std::string dir,double **W_set,const int space_num){
    string word_dir=dir+"/word/";
    std::vector<string> W_fileset=search_file(word_dir,".txt");
    int file_num=W_fileset.size();
    for(int i=0;i<file_num;i++)
        W_set[i]=new double[space_num];


    for(int i=0;i<W_fileset.size();i++){
        stringstream file;
        file << word_dir <<"word_distribution"<<i<< ".txt";
        //cout<<i<<endl;
        ifstream ifs(file.str().c_str());
        if(!ifs){
            cout<<file.str()<<" directory is not existed.";
            exit(-1);
        }
        string str;
        int j=0;
        while(getline(ifs,str)){
            string line;
            istringstream stream(str);
            
            while(getline(stream,line,',')){
                double temp=atof(line.c_str());
                W_set[i][j]=temp;
                //cout<<i<<" "<<j<<" "<<W_set[i][j]<<endl;
                j++;
            }
        }
     }
   }

void pi_read(std::string dir,double *pi_set){

        stringstream file;
        file <<dir<<"pi.csv";
        
        ifstream ifs(file.str().c_str());
        if(!ifs){
            cout<<file.str()<<" is not existed."<<endl;
            exit(-1);
        }

        string str;
        int j=0;
        while(getline(ifs,str)){
            string token;
            istringstream stream(str);
            //cout<<str<<endl;
            while(getline(stream,token,',')){
                double temp=atof(token.c_str());
                pi_set[j]=temp;
                //cout<<i<<" "<<j<<" "<<mu_set[i][j];    
            }
            j++;
        }
        //cout<<endl;


}
void color_read(int color[][3],std::string file){
        stringstream f;
        f << file;
        ifstream ifs(f.str().c_str());
        if(!ifs){
            cout<<file<<" directory is not existed.";
            exit(-1);
        }
        string str;
        int i=0;
        while(getline(ifs,str)){

            std::string line;
            std::istringstream stream(str);
            int j=0;    
                while(getline(stream,line,' ')){
                    
                    int temp=atoi(line.c_str());
                    //cout<<j<<" "<<line<<endl;
                    if(j!=0){
                        color[i][j-1]=temp;
                    }
                    j++;
            }
        i++;
        //cout<<endl;
     }
}

//**************************Reading parameter file**************************
Map read_parameter(std::string result)
{
std::string txt=result+"Parameter.txt";
std::ifstream ifs(txt.c_str());
   if(!ifs)
   {
        std::cout<<txt<<" is not existed."<<endl;
        exit(1);
    }
    int i=0;
    
    std::string str,dir;
    while(getline(ifs,str)){
            std::string token;
            std::vector<std::string> line;
            line=split(str,' ');
            //cout<<line[0]<<endl;
            if((line[0].find("データセット位置:")!=-1))
            {
             dir=line[1];
            }else if((line[0].find("Dataset:")!=-1)){
	    dir=line[1];
	    }
        i++ ;
    }
    
    dir=dir+"/map/";
    cout<<dir<<endl;
    std:vector<std::string> files_array;
    files_array=search_file(dir,".yaml");//The yaml file is searched.
    //cout<<files_array.size()<<endl;
    if(files_array.size()>1){
        cout<<"Error:There are "<<files_array.size()<<" map files!!"<<endl;
        cout<<"Remove unnecessary files!"<<endl;
        exit(-1);    
    }
    //Reading map file
    Map m;
    m=read_yaml(files_array[0]);
    m.name=dir+m.name;
    
return m;
}


//*******************Readin yaml files function*********************************

Map read_yaml(std::string yam_file){

    Map m;
    std::ifstream ifs(yam_file.c_str());
    int i=0;
    
    std::string str,dir;
    while(getline(ifs,str))
        {   
            std::string token;
            //std::stringstream stream(str);
            std::vector<std::string> line;
            //cout<<"a"<<endl;
            line=split(str,' ');
            if(line.size()==0){
                break;
            }else if(line[0]=="image:"){ 
                m.name=line[1];
            }else if(line[0]=="origin:")
            {
                std::vector<std::string> x;
                x=split(line[1],'[');
                m.coordinate[0]=atof(x[1].c_str());
                m.coordinate[1]=atof(line[2].c_str());
                m.coordinate[2]=0;
            }

        i++ ;
    }
    return m;
}
