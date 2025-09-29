![Problem (unicode1)](image.png)  
a) '\x00'  
b) __repr__() is a special function in a class to return a user defined string while directly call print(obj), the __str()__ is used  
c)  
![answer for Q_c](image-1.png)  
Seems chr(0) is \x00 in string but will print nothing when call function print.  

![Problem (unicode2)](image-2.png)  
a)  
![print for different utf](image-3.png)  
The output is more readable in utf-8 while the input is within ASCII.  
![mem used](image-4.png)  
The utf-8 will use less mem.  
b)Not all character is encoded into one byte. So decode byte by byte can cause crash like:  
![crash with simplified Chinese](image-5.png)  
c) b'\xc0\x80' this is valid in all three encoding format.  

