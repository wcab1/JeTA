# JeTA
*Jellyfish Track and Analysis*
Image Stabilization developed by William Boyd
Original Pulse detetcion scripts devloped by Gradinaru Lab at Caltech and modified by William Boyd
---------------------------------------
*Requirements*
Yolov5 - jellyfish weight included, 
Matplotlib (may need to install tkinter as well), 
NumPy, 
Pandas, 
opencv, 
PyTorch, 
Skimage, 
---------------------------------------


                                                                                              
                                          ▒▒▒▒▒▒▒▒▒▒▒▒▒▒                                      
                                        ▒▒▒▒▒▒░░░░░░░░▒▒▒▒                                    
                                      ▒▒▒▒░░░░░░░░░░░░░░▒▒▒▒                                  
                                    ▒▒▒▒░░▒▒▒▒░░░░░░░░▒▒░░▒▒▒▒                                
                                    ▒▒▒▒▒▒░░░░  ▒▒░░░░░░▒▒▒▒▒▒                                
                                    ▒▒▒▒▒▒░░░░░░░░░░░░░░  ░░▒▒                                
                                  ▒▒▒▒░░░░▒▒░░░░░░▒▒▒▒▒▒  ▒▒▒▒▒▒                              
                                ▒▒▒▒▒▒░░░░▒▒  ░░░░▒▒░░▒▒▒▒░░▒▒▒▒▒▒                            
                                ▒▒▒▒▒▒▒▒░░░░▒▒▒▒░░░░▒▒░░▒▒  ▒▒▒▒▒▒                            
                                ▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░▒▒░░░░▒▒▒▒▒▒▒▒                            
                              ░░░░    ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ░░  ░░                          
                              ░░  ░░    ░░  ▒▒    ▒▒▒▒  ░░    ░░    ░░                        
                          ░░░░    ░░  ░░░░  ▒▒    ▒▒░░    ░░  ░░    ░░                        
                        ░░░░    ░░    ░░  ░░░░    ▒▒░░    ░░  ░░░░  ░░░░                      
                        ░░    ░░    ░░    ░░▒▒    ▒▒░░░░  ░░    ░░    ░░                      
                        ░░    ░░    ░░      ▒▒    ▒▒░░▒▒  ░░      ░░  ░░                      
                        ░░    ░░    ░░      ░░░░    ░░    ░░      ░░  ░░                      
                        ░░    ░░    ░░      ░░░░    ░░    ░░      ░░  ░░                      
                        ░░    ░░    ░░      ▒▒░░    ▒▒    ░░      ░░  ░░                      
                        ░░    ░░    ░░      ░░░░    ▒▒▒▒    ░░    ░░  ░░                      
                        ░░    ░░    ░░    ░░░░░░    ░░▒▒    ░░    ░░  ░░                      
                        ░░    ░░    ░░    ░░░░▒▒    ▒▒░░    ░░    ░░  ░░                      
                        ░░    ░░    ░░    ▒▒░░▒▒    ▒▒░░    ░░    ░░    ░░░░░░                
                              ░░    ░░      ▒▒░░    ▒▒░░    ░░    ░░          ░░              
                  ░░          ░░    ░░      ▒▒      ▒▒      ░░    ░░          ░░              
                              ░░    ░░      ░░░░      ░░    ░░    ░░          ░░              
                          ░░░░      ░░        ▒▒      ▒▒    ░░    ░░          ░░              
                          ░░        ░░      ▒▒▒▒    ░░▒▒    ░░    ░░                          
                          ░░          ░░    ░░░░      ▒▒    ░░      ░░                        
                          ░░          ░░    ▒▒░░    ▒▒░░      ░░    ░░                        
                            ░░        ░░    ▒▒░░    ▒▒░░      ░░    ░░                        
                            ░░      ░░        ▒▒    ░░        ░░    ░░                        
                            ░░    ░░          ▒▒      ░░    ░░    ░░                          
                                  ░░          ░░      ░░░░  ░░                                
                                  ░░        ▒▒        ░░▒▒  ░░                                
                                  ░░        ▒▒        ▒▒      ░░                              
                                    ░░    ░░          ▒▒░░    ░░                              
                                    ░░                  ░░    ░░                              
                                    ░░                                                        
                                    ░░                                                        
                                                                                              
                                                                                              
                                                                                              
                                                                                              
                                                                                              
                                                                                              
                                                                                              
                                                                                            
