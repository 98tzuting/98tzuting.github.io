program ifElseIfElseProg
implicit none

   ! local variable declaration
   integer :: a = 100
 
   ! check the logical condition using if statement
   if( a == 10 ) then
  
      ! if condition is true then print the following 
      print*, "Value of a is 10" 
   
   else if( a == 20 ) then
  
      ! if else if condition is true 
      print*, "Value of a is 20" 
  
   else if( a == 30 ) then
   
      ! if else if condition is true  
      print*, "Value of a is 30" 
  
   else
   
      ! if none of the conditions is true 
      print*, "None of the values is matching" 
      
   end if
   
   print*, "exact value of a is ", a
 
end program ifElseIfElseProg