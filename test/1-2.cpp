#include<iostream>
#include<cstdlib>
using namespace std;
int main(int argc, char *argv[])
{
    float Vcc, Ie, beta, Vb, Vce, Rb2, Rb1, Vbe, Ve, Re, Ic, Vc, RL;
    cout << "step 1:Please input thee value of Vcc" << endl;
    cin >> Vcc;
    cout << "step 2:Please input thee value of Ie" << endl;
    cin >> Ie;
    cout << "step 3:Please input thee value of beta" << endl;
    cin >> beta;
    Vb = Vcc/3;
    cout << "the value of Vb is equal to : " << Vb << endl;
    Vce = Vcc/3;
    cout << "the value of Vce is equal to : " << Vce << endl;
    Rb2 = (Vcc/0.1*Ie)/3;
    cout << "the value of Rb2 is equal to : " << Rb2 << endl;
    Rb1 = 2*Rb2;
    cout << "the value of Rb1 is equal to : " << Rb1 << endl;
    Vbe = 0.7; Ve =Vb - Vbe; Re = Ve/Ie;
    cout << "Re equal to : " << Re << endl;
    Ic = (beta/(beta+1))*Ie; Vc = 2*Vcc/3; RL = (Vcc-Vc)/Ic;
    cout << "RL equal to : " << RL <<endl;
    system("pause");

}