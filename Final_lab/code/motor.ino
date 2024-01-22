#include <Wire.h>
#include <Adafruit_MotorShield.h>

// Adafruit_MotorShield object creation
Adafruit_MotorShield AFMS = Adafruit_MotorShield(); 

// connect DC motor to M4 Port
Adafruit_DCMotor *myMotor = AFMS.getMotor(4);
int speed = 0;

void setup() {
  Serial.begin(9600);  //Begin Serial communication

  // Adafruit Motor Shield Set
  AFMS.begin(); 

  // DC motor basic velocity set
  myMotor->setSpeed(100); 
}

void loop() {
  if (Serial.available()) {  //Check the data exists in serial Port
    String command = Serial.readStringUntil('\n');  // Read data
    // Convert strings to array and save array
    int array[2];
    int index = 0;
    for (int i = 0; i < command.length(); i++) {
        int commaIndex = command.indexOf(',', i);
        if (commaIndex == -1) {
            array[index] = command.substring(i).toInt();
            break;
        } else {
            array[index] = command.substring(i, commaIndex).toInt();
            index++;
            i = commaIndex;
        }
    }
    int direction = array[0]; // direction command from server.py
    int velocity =  array[1]; // velocity command from server.py
      
    if( direction != 0){    
      myMotor->setSpeed(velocity);
      if(direction == 1){
        myMotor->run(BACKWARD);
      }else{
        myMotor->run(FORWARD);
      }
    }else{
      myMotor->run(RELEASE); 
    }
  }
}