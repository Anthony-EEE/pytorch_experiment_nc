// Copyright Tianchen Shen, Irene Di Giulio, Matthew Howard. 
// This code is provided confidentially for purposes of peer review only. 
// All rights reserved.


// PWM is connected to pin 3.
const int pinPwm = 3;

// DIR is connected to pin 2.
const int pinDir = 2;

void setup() {                
  pinMode(pinPwm, OUTPUT); // Set the PWM pin as an output
  pinMode(pinDir, OUTPUT); // Set the DIR pin as an output
}

void loop() {
  static int count = 0; 

  if (count < 50) { /
    analogWrite(pinPwm, 55);   // Set PWM to change the frequency of the Scotch yoke
    delay(8000);               
    analogWrite(pinPwm, 0);    

    count++; 
  }

}
