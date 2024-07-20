import numpy as np
from math import *
import neuralfoil as nf
import aerosandbox as asb
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras
import time

warnings.filterwarnings("ignore")
model = keras.models.load_model("C:/Users/kaust/Downloads/PSOmodel.keras")

def cl_cd_func(wing_area, aspect_ratio, e, airfoil, weight, kinematic_viscosity, density):

    global model

    chord = (wing_area/aspect_ratio) ** 0.5
    curr_CLCD = 0 
    curr_CL = 0
    CLCD_max_stats = []
    CL_max_stats = []

    e = 1.78*(1-(0.045*(aspect_ratio**0.68)))-0.64
    
    aoa = np.linspace(0, 40, num=41)
    re = np.linspace(0, (40 * chord/kinematic_viscosity), num=41)
    AOA, RE = np.meshgrid(aoa, re)

    aero = nf.get_aero_from_airfoil(
        airfoil = asb.Airfoil(airfoil),
        alpha = AOA.flatten(), Re=RE.flatten(),
    )    

    Cl = aero["CL"]
    Cd = aero["CD"]

    inputs = np.array([[wing_area, aspect_ratio, (i%41)/2, (i//41)/2, Cl[i]] for i in range(len(Cl))])

    CL = model.predict(np.array(inputs),verbose = 0)
    lift = CL * 0.5 * density * wing_area

    for i in range(41):
        aoa = i/2
        for j in range(41):
            velocity = j/2
            if lift[i + j*41]*(velocity**2) > weight:
                CD = Cd[i + j*41] + ((CL[i + j*41]**2)/(3.1415*e*aspect_ratio))
                if (CL[i + j*41]/CD) > curr_CLCD:
                    curr_CLCD = CL[i + j*41]/CD
                    CLCD_max_stats = [j/2, velocity, CL[i + j*41], CD, CL[i + j*41]/CD]
                if CL[i + j*41] > curr_CL:
                    curr_CL = CL[i + j*41]
                    CL_max_stats = [j/2, velocity, CL[i + j*41], CD, CL[i + j*41]/CD]
                break

    return CLCD_max_stats[2], CLCD_max_stats[3], CL_max_stats[2], CL_max_stats[3]
    
def takeoff_thrust_func(CL, CD, friction_rolling, takeoff_distance, density, wing_area, weight):
    g = 9.80665
    stall_velocity = sqrt((2*weight)/(wing_area*density*CL))
    effective_stall_velocity = 1.2 * stall_velocity
    #print(effective_stall_velocity)
    B = (g*density*wing_area*(CD-friction_rolling*CL))/(2*weight)
    thrust = ((weight/g)*(exp(takeoff_distance * 2 * B)) * B * (effective_stall_velocity**2))/((exp(takeoff_distance * 2 * B))-1) + friction_rolling * weight
    return thrust, effective_stall_velocity

def takeoff_time_func(CL, CD, friction_rolling, thrust, density, wing_area, weight):
    g = 9.80665
    stall_velocity = sqrt((2*weight)/(wing_area*density*CL))
    effective_stall_velocity = 1.2 * stall_velocity
    B = (g*density*wing_area*(CD-friction_rolling*CL))/(2*weight)
    A = (thrust - friction_rolling * weight)/(weight/g)
    takeoff_time = (1/(2*sqrt(A*B))) * (log((sqrt(A) + effective_stall_velocity * sqrt(B))/(sqrt(A) - effective_stall_velocity * sqrt(B))))
    return takeoff_time

def straight_func(CL, CD, density, wing_area, weight):
    straight_velocity = sqrt((2*weight)/(CL*density*wing_area))
    return straight_velocity

def turning_func(CL, CD, thrust, density, wing_area, weight):
    g = 9.80665
    bank_angle = min(7.5  * pi/18, acos(weight * CD/(thrust * CL)))
    thrust = (weight * CD)/(cos(bank_angle) * CL) 
    turning_radius = (2*weight)/(density * CL * wing_area * sin(bank_angle) * g)
    turning_velocity = sqrt((2*thrust)/(density*CD*wing_area))
    return turning_radius, turning_velocity, thrust, bank_angle
    
def scorefunc(mass, prop_mass, airfoil, wing_area, aspect_ratio):

        wing_area = float(wing_area)
        aspect_ratio = float(aspect_ratio)
        #print(wing_area, aspect_ratio)
        #wing_area = 0.45
        #aspect_ratio = 5.75

        #physics
        g = 9.80665
        density = 1.225
        friction_rolling = 0.025
        kinematic_viscosity = 0.000014207

        #lap details
        takeoff_distance = 15
        straight_distance = 304.8
        altitude = 7.6

        #plane details
        e = 0.8
        battery_efficiency = 0.95

        #constraints
        max_battery_size = 360000

        #weight
        total_mass = mass + prop_mass
        weight = total_mass * g

        #clcd
        CL, CD, CLSTALL, CDSTALL = cl_cd_func(wing_area, aspect_ratio, e, airfoil, weight, kinematic_viscosity, density)   
        #print("CLCD")
        #print(CL, CD, CLSTALL, CDSTALL)

        #max thrust calculations
        max_thrust, effective_stall_velocity = takeoff_thrust_func(CLSTALL, CDSTALL, friction_rolling, takeoff_distance, density, wing_area, weight)

        #flight calculations
        straight_velocity = straight_func(CL, CD, density, wing_area, weight)
        takeoff_time = takeoff_time_func(CLSTALL, CDSTALL, friction_rolling, max_thrust, density, wing_area, weight)
        turning_radius, turning_velocity, turning_thrust, bank_angle = turning_func(CLSTALL, CDSTALL, max_thrust, density, wing_area, weight)
        time_straight = (3 * straight_distance * 2 - takeoff_distance)/straight_velocity
        time_turn = (3 * 4 * pi * turning_radius)/turning_velocity
        time = time_straight + takeoff_time + time_turn
        straight_thrust = (weight * CD)/CL
        max_speed = max(straight_velocity, turning_velocity)
        #print("speeds")
        #print(straight_velocity, turning_velocity)

        #motor calculcations
        power = max_thrust * max_speed
        mass_motor = power/3000
        #print("power")
        #print(power, mass_motor)

        #battery calculations
        #print("energy")
        energy = (1/(0.86*0.85)) * power * time
        mass_battery = energy/(125*3600)
        #print(energy, mass_battery)

        return time, mass_battery+mass_motor

def cost_func(a=4, b=4, cd=12, wing_area=0.4, AR=7):
        airfoil = "naca"+str(a)+str(b)+str(cd)
        mass = 1.5
        prop_mass = 0.0001
        while True:
                time, newmass = scorefunc(mass, prop_mass, airfoil, wing_area, AR)
                #print("time")
                #print(time, newmass)
                #print("\n\n")
                if newmass/prop_mass < 1.01:
                        break
                prop_mass = newmass      

        #print(mass + prop_mass)
        return (mass + prop_mass)

if __name__ == '__main__':
    while True:
        if os.path.isfile("C:/Users/kaust/Downloads/PSOmatlab.txt"):
            time.sleep(0.1)
            f = open("C:/Users/kaust/Downloads/PSOmatlab.txt", "r")
            lines = f.readlines()
            lines = [i[:-1] for i in lines]
            print(lines)
            f.close()
            os.remove("C:/Users/kaust/Downloads/PSOmatlab.txt")
            result = cost_func(lines[0], lines[1], lines[2], lines[3], lines[4])[0]
            f2 = open("C:/Users/kaust/Downloads/PSOmatlab_return.txt", "w")
            f2.write(str(result))
            f2.close()