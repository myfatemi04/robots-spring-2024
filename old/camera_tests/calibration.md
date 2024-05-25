These cameras use the Brown-Conrady method of calibration.
These are stored in the `calibration` folder.

To see more about the order of parameters, see: https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/k4atypes_8h_source.html

To have a second copy of what these mean:
```c
 typedef union
 {
     struct _param
     {
         float cx;            
         float cy;            
         float fx;            
         float fy;            
         float k1;            
         float k2;            
         float k3;            
         float k4;            
         float k5;            
         float k6;            
         float codx;          
         float cody;          
         float p2;            
         float p1;            
         float metric_radius; 
     } param;                 
     float v[15];             
 } k4a_calibration_intrinsic_parameters_t;
```
