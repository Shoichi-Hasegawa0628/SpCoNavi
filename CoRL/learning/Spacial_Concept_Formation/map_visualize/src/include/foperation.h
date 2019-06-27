
typedef struct Map{
    double coordinate[3];
    std::string name;
}Map;

//***********************Search file function**************************
std::vector<std::string> search_file(std::string dir,std::string extension);
//*******************Spliting function*********************************
std::vector<std::string>split(const std::string &str,char sep);
//**************************Reading files that describe average of gaussian.*********************
void mu_read(std::string mu_dir,double mu_set[][4]);

void  two_mu_read(std::string mu_dir,double mu_set[][2]);

void  sigma_read(std::string dir,double sigma_set[][4][4]);

void  two_sigma_read(std::string dir,double sigma_set[][2][2]);

void word_read(std::string dir,double **W_set,const int space_num);

void pi_read(std::string dir,double *pi_set);

void color_read(int color[][3],std::string file);


Map read_yaml(std::string yam_file);

Map read_parameter(std::string result);


