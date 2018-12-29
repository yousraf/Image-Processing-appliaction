
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

void help()
{
    cout << "\nThis program demonstrates line finding with the Hough transform.\n"
    "Usage:\n"
    "./houghlines <image_name>, Default is pic1.jpg\n" << endl;
}

int main(int argc, char** argv)
{
    const char* filename = argc >= 2 ? argv[1] : "pic1.jpg";
    
    Mat src = imread("/Users/rababboulouchgour/Downloads/CHykbDHW8AE_3KZ.jpg", 0);
    if(src.empty())
    {
        help();
        cout << "can not open " << filename << endl;
        return -1;
    }
    
    Mat dst, cdst;
    Canny(src, dst, 50, 200, 3);
    cvtColor(dst, cdst, CV_GRAY2BGR);
    
//#if 0
    vector<Vec2f> lines;
    HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );
    
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
    }
/*#else
    vector<Vec4i> lines;
    HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }
//#endif */
    imshow("source", src);
    imshow("detected lines", cdst);
    
    waitKey();
    
    return 0;
}

/*#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat img;
int threshval = 100;

static void on_trackbar(int, void*)
{
    Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
    Mat labelImage(img.size(), CV_32S);
    int nLabels = connectedComponents(bw, labelImage, 8);
    std::vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0);//background
    for(int label = 1; label < nLabels; ++label){
        colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    Mat dst(img.size(), CV_8UC3);
    for(int r = 0; r < dst.rows; ++r){
        for(int c = 0; c < dst.cols; ++c){
            int label = labelImage.at<int>(r, c);
            Vec3b &pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
        }
    }
    
    imshow( "Connected Components", dst );
}

static void help()
{
    cout << "\n This program demonstrates connected components and use of the trackbar\n"
    "Usage: \n"
    "  ./connected_components <image(../data/stuff.jpg as default)>\n"
    "The image is converted to grayscale and displayed, another image has a trackbar\n"
    "that controls thresholding and thereby the extracted contours which are drawn in color\n";
}

const char* keys =
{
    "{help h||}{@image|../data/stuff.jpg|image for converting to a grayscale}"
};

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    string inputImage = parser.get<string>(0);
    img = imread("/Users/rababboulouchgour/Downloads/CHykbDHW8AE_3KZ.jpg", 0);
    
    if(img.empty())
    {
        cout << "Could not read input image file: " << inputImage << endl;
        return -1;
    }
    
    namedWindow( "Image", 1 );
    imshow( "Image", img );
    
    namedWindow( "Connected Components", 1 );
    createTrackbar( "Threshold", "Connected Components", &threshval, 255, on_trackbar );
    on_trackbar(threshval, 0);
    
    waitKey(0);
    return 0;
}*/

/*#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <iterator>
#include "norm_0_255.cpp"   //dans la doc
#include <math.h>
#define racine(a, n) pow((a), 1.0/(n))
using namespace cv;
using namespace std;
void bgrtogray(Mat source,Mat dest){
    int i,j;
    for(i=0;i<source.rows;i++)
    {
        for(j=0;j<source.cols; j++){
            Vec3b pixel=source.at<Vec3b>(i,j);
            uchar blue=pixel.val[0];
            uchar green=pixel.val[1];
            uchar red=pixel.val[2];
            dest.at<uchar>(i,j)=(blue+red+green)/3;
        }
    }
}

bool seg(Mat image,set<int[2]> visited,set<int[2]> next,vector<int[2]> region){
    bool nextregion=true;
    set<int[2]>::iterator it=next.begin();
    int x=(*it)[0];
    int y=(*it)[1];
    if(!visited.count({x,y})){
        next.erase({x,y});
        if(image.at<uchar>(x,y)==image.at<uchar>(x+1,y)){
            region.push_back({x+1,y});
            nextregion=false;
        }else{
            next.insert({x+1,y});
        }
        if(image.at<uchar>(x,y)==image.at<uchar>(x,y+1)){
            region.push_back({x,y+1});
            nextregion=false;
        }else{
            next.insert({x,y+1});
        }
        if(image.at<uchar>(x,y)==image.at<uchar>(x-1,y)){
            region.push_back({x-1,y});
            nextregion=false;
        }else{
            next.insert({x-1,y});
        }
        if(image.at<uchar>(x,y)==image.at<uchar>(x,y-1)){
            region.push_back({x,y-1});
            nextregion=false;
        }else{
            next.insert({x,y-1});
        }
        if(!nextregion)
        {visited.insert(*next.begin());}
    }
    return nextregion;
}*/
/*
void  region(Mat image,vector<vector<int[2]>> regions){
    set<int[2]> visited,next;
    next.insert({0,0}); int i=0;
    while(visited.size()<image.rows*image.cols){
       if(seg(image,visited,next,regions[i]))  i++;
    }
}*/

    /* Mat dest;
    vector<int[2]> visited;
    vector<int[2]> next;
    vector<vector<int[2]>> regions;
    regions[0][0][0]=0;
    regions[0][0][1]=0;
    int regcount=0;
    Vec3b pixel;
    int i=1;
    int j=1;
    int elm=0;
    int elm2=0;
    int now=0;
    int now2=0;
    while(i<dest.rows-1 && j<dest.cols-1){
        bool r=false;
        if(dest.at<uchar>(i,j)== dest.at<uchar>(i+1,j)){
            regions[regcount][elm][0]=i+1;
            regions[regcount][elm][1]=j;
            r=true;
            elm++;
        }else{
            
            next[elm2][0]=i+1;
            next[elm2][1]=j;
            elm2++;
        }
        if(dest.at<uchar>(i,j)== dest.at<uchar>(i-1,j)){
            regions[regcount][elm][0]=i-1;
            regions[regcount][elm][1]=j;
            r=true;

            elm++;
        }else{
            next[elm2][0]=i+1;
            next[elm2][1]=j;
            elm2++;
        }
        if(dest.at<uchar>(i,j)== dest.at<uchar>(i,j+1)){
            regions[regcount][elm][0]=i;
            regions[regcount][elm][1]=j+1;
            r=true;

            elm++;
        }else {
            next[elm2][0]=i;
            next[elm2][1]=j+1;
            elm2++;
        }
        if(dest.at<uchar>(i,j)== dest.at<uchar>(i,j-1)){
            regions[regcount][elm][0]=i;
            regions[regcount][elm][1]=j-1;
            r=true;
            elm++;
        }else{
            next[elm2][0]=i;
            next[elm2][1]=j-1;
            elm2++;
        }
        if(r){
            now++;
            i=regions[regcount][now][0];
            j=regions[regcount][now][1];
            
        }else{
            regcount ++;
            now=0;
            i=next[now2][0];
            j=next[now2][1];
            now2++;
        }
    }
Mat getSkin(Mat &img){
    Vec3b cwhite = Vec3b::all(255);
    Vec3b cblack = Vec3b::all(0);
    Mat skin=img.clone();
    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            
            Vec3b pix_bgr = img.ptr<Vec3b>(i)[j];
            int B = pix_bgr.val[0];
            int G = pix_bgr.val[1];
            int R = pix_bgr.val[2];
            // apply rgb rule
            bool tst = ((R>95) && (G>40) && (B>20) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B))||((R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B));
            if(!tst){
                skin.ptr<Vec3b>(i)[j] = cblack; }
            
                else{
                    skin.ptr<Vec3b>(i)[j] = cwhite;
                    
                }
        }
    }
    return skin;

}
void video(){
    Mat show;
    VideoCapture cam(0);
    if(!cam.isOpened())
    {
        return;
    }
    int frame_width=   cam.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height=   cam.get(CV_CAP_PROP_FRAME_HEIGHT);
    VideoWriter video("out.avi",0,10, Size(frame_width,frame_height),true);
    for(;;){
        Mat skinframe;
        Mat frame;
        cam >> frame;
        skinframe=getSkin(frame);
        video.write(skinframe);
        imshow( "Frame",skinframe);
        char c = (char)waitKey(33);
        if( c == 27 ) break;
    }
    cam.release();
    cout << video.get(CAP_PROP_FORMAT) ;
    return;
}
void afficherHist(int hist[3][256])
{
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<256;j++) {
            cout<<hist[i][j]<<endl;
        }
        cout << "----------------------------" <<endl;
    }}

void histogram(Mat image, int hist[3][256]){
        int i,j;
        for(i=0;i<image.rows;i++)
        {
            for(j=0;j<image.cols;j++){
                Vec3b pixel=image.at<Vec3b>(i,j);
                uchar blue=pixel.val[0];
                uchar green=pixel.val[1];
                uchar red=pixel.val[2];
                
                hist[2][(int)red]=hist[2][(int)red]+1;
                hist[1][(int)green]=hist[1][(int)green]+1;
                hist[0][(int)blue]=hist[0][(int)blue]+1;
            }
        }}

int main(int argc, const char * argv[]) {
    
    namedWindow("image couleur");
    Mat source =imread("/Users/rababboulouchgour/Desktop/037.jpg",1); //Mat   Mat dest;
    if(source.empty()) //Tester si on a pu lire ou pas
    {
        cout <<"impossible de charger l'image"<<endl; //endl pour formatage, retour à la ligne .. etc
        return -1;
    }
    //affichage del'image
    namedWindow("image source"); // CRéation d'une fenêtre qui sera appelée : Image source
    imshow("image source",source); //Affichage de l'image SOURCE dans la fenêtre image source
    waitKey(0);
    cvtColor(source,source, CV_RGB2GRAY);
    namedWindow("image en niveaux de gris");
    imshow("image en niveaux de gris",source);
    waitKey(0);
    Mat noise = Mat(source.size(),CV_64F);
    normalize(source,source, 0.0, 1.0, CV_MINMAX, CV_64F);
    randn(noise, 0, 0.1);
    source =source + noise;
    namedWindow("image bruitée");
    imshow("image bruitée",source);
    waitKey(0);
    // source=norm_0_255(source);  //I have to include SOURCE
    //result.convertTo(result, CV_32F, 255, 0);
    Mat dest;
    dest.create(source.rows,source.cols, CV_8UC1);
    if(source.empty())
    {
        cout <<"impossible de charger l'image"<<endl; //endl pour formatage, retour à la ligne .. etc
        return -1;
    }
    for (int i=0;i<source.rows-1; i++)
    {
        for (int j=0;j<source.cols-1; j++){
            if(i==0){
                if(j==0){
                    dest.at<uchar>(i,j)=(source.at<uchar>(i,j)+source.at<uchar>(i+1,j)+source.at<uchar>(i,j+1)+source.at<uchar>(i+1,j+1))/9;
                } else if(j==source.cols-1){
                    dest.at<uchar>(i,j)=(source.at<uchar>(i,j)+source.at<uchar>(i,j-1)+source.at<uchar>(i+1,j)+source.at<uchar>(i+1,j-1))/9;
                } else {
                    dest.at<uchar>(i,j)=(source.at<uchar>(i,j)+source.at<uchar>(i,j-1)+source.at<uchar>(i+1,j)+source.at<uchar>(i+1,j-1)+source.at<uchar>(i,j+1)+source.at<uchar>(i+1,j+1))/9;
                }
            } else if(i==source.rows-1){
                if(j==0){
                    dest.at<uchar>(i,j)=(source.at<uchar>(i,j)+source.at<uchar>(i-1,j)+source.at<uchar>(i-1,j+1)+source.at<uchar>(i,j+1))/9;
                } else if(j==source.cols-1){
                    dest.at<uchar>(i,j)=(source.at<uchar>(i,j)+source.at<uchar>(i-1,j-1)+source.at<uchar>(i-1,j)+source.at<uchar>(i,j-1))/9;
                } else {
                    dest.at<uchar>(i,j)=(source.at<uchar>(i,j)+source.at<uchar>(i-1,j-1)+source.at<uchar>(i-1,j)+source.at<uchar>(i-1,j+1)+source.at<uchar>(i,j-1)+source.at<uchar>(i,j+1))/9;
                }
            } else {
                if(j==0){
                    dest.at<uchar>(i,j)=(source.at<uchar>(i,j)+source.at<uchar>(i-1,j)+source.at<uchar>(i-1,j+1)+source.at<uchar>(i+1,j)+source.at<uchar>(i,j+1)+source.at<uchar>(i+1,j+1))/9;
                    
                } else if(j==source.cols-1){
                    dest.at<uchar>(i,j)=(source.at<uchar>(i,j)+source.at<uchar>(i-1,j-1)+source.at<uchar>(i-1,j)+source.at<uchar>(i,j-1)+source.at<uchar>(i+1,j)+source.at<uchar>(i+1,j-1))/9;
                    
                } else {
                    dest.at<uchar>(i,j)=(source.at<uchar>(i,j)+source.at<uchar>(i-1,j-1)+source.at<uchar>(i-1,j)+source.at<uchar>(i-1,j+1)+source.at<uchar>(i,j-1)+source.at<uchar>(i+1,j)+source.at<uchar>(i+1,j-1)+source.at<uchar>(i,j+1)+source.at<uchar>(i+1,j+1))/9;
                    
                }
            }
            
        }
    }
    namedWindow("image  après filtre");
    imshow("image après filtre",dest);
    
    
    
    video();
    
    return 0;
}*/
