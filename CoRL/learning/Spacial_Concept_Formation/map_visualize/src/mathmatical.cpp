#include "include/mathmatic.h"
#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
//==========saras=============
double TwoD_saras(double **matrix){
 double det=0;
 det +=matrix[0][0]*matrix[1][1];
 det -=matrix[0][1]*matrix[1][0];
 return det;
}
//==========matrix check=============

void matrix_chek(double **matrix,const int n){
    cout<<"-----"<<n<<"×"<<n<<"-----"<<endl;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
                cout<<" "<<matrix[i][j];
            }
            cout<<"\n";
        }
}
//==========vector check=============
void vector_check(double *v,const int n){
    for(int i=0;i<n;i++){
        cout<<" "<<v[i];
    }
    cout<<endl;
}
//==========inv matrix check=============
void inv_matirx(double** matrix,const int n){
    double inv_sigma[n][n]; 
    double buf; 

    //---Make unit matrix
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            inv_sigma[i][j]=(i==j)?1.0:0.0;
        }
    }

    //---Sweep out 
    for(int i=0;i<n;i++){
        buf=1.0/matrix[i][i];
        for(int j=0;j<n;j++){
            matrix[i][j]*=buf;
            inv_sigma[i][j]*=buf;
        }

        for(int j=0;j<n;j++){
            if(i!=j){
                buf=matrix[j][i];
                for(int k=0;k<n;k++){
                    matrix[j][k]-=matrix[i][k]*buf;
                    inv_sigma[j][k]-=inv_sigma[i][k]*buf;
                }
            }
        }
    }
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            matrix[i][j]=inv_sigma[i][j];
        }
    }
}

//==============make zero array==========================
void zero_array(double array[],int n){
    
    for(int i=0;i<n;i++){
    array[i]=0.0;
    //cout<<array[i]<<endl;
    }

}
//=========caluculate argmax==============================
int argmax(double array[],const int n){
int ans=0;
for(int i=0;i<n;i++){
    if(array[ans]<array[i]){
        ans=i;
        }
    }
return ans;
}
//=========copying array==============================
void cpy_array(double new_array[],const double data[]){
    int i=0;
    while(data[i]!='\0'){
        double temp;
        temp=data[i];
        new_array[i]=temp;
        //cout<<temp<<endl;;
        i++;
    }

}
//==========sumation elements of array========================
double sum_array(double array[],const int n){
    double ans=0;
    for(int i=0;i<n;i++){
        ans+=array[i];
    }
    return ans;
}

//=========cofactor=============
double ND_cofactor(double** matrix,const int n){
    double det=0;
    int i=0;
        //matrix_chek(matrix,n);
        if(n<0){
        cout<<"Error:Second argument shoudl be more than zero; "<<endl;
        exit(-1);
         }

        for(int row=0;row<n;row++){
            double *temp[n-1];
            int l=0;
            for(int i=0;i<n;i++){
                if(row!=i){
                        temp[l]=&matrix[i][1];
                        l++;
                    }
                }
            double k=pow(-1,row);
            if(n==3){
            det +=k*matrix[row][0]*TwoD_saras(temp);
            }else if(n>3){
            double det_temp;
            det_temp=k*matrix[row][0]*ND_cofactor(temp,n-1);
            
            det+=det_temp;
        }
    }
return det;
}
