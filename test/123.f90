PROGRAM homework_2
IMPLICIT NONE
REAL::f,t
WRITE(*,*)'Enter the t:'
READ(*,*)t
IF(t>=0)THEN
f=-3.*t**2+5.
ELSE
f=3.*t**2+5.
END IF
WRITE(*,*)'f=',f
END