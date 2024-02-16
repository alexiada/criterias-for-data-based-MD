import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys, time
from celluloid import Camera
import numba
#import ann_md

rng = np.random.default_rng()
LX = 5. # BOX WIDTH
LY = 5.  # BOX HEIGHT
BOX = np.array([LX, LY],dtype="float64")
DENSITY = 0.5
NUM_PARTICLES = int(LX*LY*DENSITY)
MAX_NEIGH = 40
CUT_OFF = 2.5 #2.**(1./6.)#2.5
CUT_OFF_2 = CUT_OFF**2
SHELL = 0.3*CUT_OFF
SHELL_2 = SHELL**2
CUT_OFF_LIST = CUT_OFF + SHELL
CUT_OFF_LIST_2 = CUT_OFF_LIST**2
###########################
TEMPERATURE = 1.
VEL = np.sqrt(2.* (1.-1./NUM_PARTICLES)*TEMPERATURE)
DT = 0.005
N_STEPS = 1000
N_OUT   = 1
###########################
#LJ UNITS: MASS = 1., EPS = 1., SIGMA = 1.

class MD():
    def __init__(self):
        self.pos = np.zeros((NUM_PARTICLES, 2), dtype="float64")
        self.pos_0 = np.zeros((NUM_PARTICLES, 2), dtype="float64")
        self.vel = np.zeros((NUM_PARTICLES, 2), dtype="float64")
        self.acc = np.zeros((NUM_PARTICLES, 2), dtype="float64")
        DX = LX/np.sqrt(NUM_PARTICLES)
        DY = LY/np.sqrt(NUM_PARTICLES)
        XY = np.float64((np.mgrid[0:LX:DX, 0:LY:DY]).reshape(2,-1).T)
        self.pos = XY[:NUM_PARTICLES,:]
        self.pos_0 = np.copy(self.pos)
        self.vel = rng.uniform(-1, 1, (NUM_PARTICLES, 2))
        shift =np.sum(self.vel)/NUM_PARTICLES
        self.vel -= shift 
        self.rescale_vel()
        self.neig_list = build_neighbor_list(self.pos)
        self.virSum = 0.
        self.uSum = 0.
        self.list_updates = 0
        self.trajectory_pos = []
        self.trajectory_vel = []
        self.trajectory_acc = []

    def update_pos(self):
        self.vel += 0.5*DT*self.acc
        self.pos += DT*self.vel
        self.pos = self.pos % BOX # apply periodic bc
        
    def update_vel(self):
        self.vel += 0.5*DT*self.acc        

    def calc_properties(self, step):
        v2Sum = np.sum(self.vel**2)
        kinEnr = 0.5*v2Sum/NUM_PARTICLES
        potEnr = self.uSum/NUM_PARTICLES 
        press = 0.5*(self.virSum+v2Sum)/NUM_PARTICLES
        print("step ", step, ": K =", kinEnr, ", U =", potEnr, ", P =", press)
        self.virSum = 0.
        self.uSum = 0.

    def append_trajectory(self):
        self.trajectory_pos.append(self.pos.copy())
        self.trajectory_vel.append(self.vel.copy())
        self.trajectory_acc.append(self.acc.copy())
        
    def rescale_vel(self):
        v2_ave = np.sum(self.vel**2)/NUM_PARTICLES
        scale_fact = VEL/np.sqrt(v2_ave)
        self.vel *= scale_fact      
        
    def update_list(self):
        delta = abs(self.pos - self.pos_0)
        delta[:,0]=np.minimum(delta[:,0], LX - delta[:,0]) 
        delta[:,1]=np.minimum(delta[:,1], LY - delta[:,1]) 
        max_delta_sq = np.max(delta[:,0]**2+delta[:,1]**2)
        if max_delta_sq > 0.25*SHELL_2:
            self.neig_list = build_neighbor_list(self.pos)
            self.pos_0 = np.copy(self.pos)
            self.list_updates += 1
  
    def single_step(self):
        self.update_pos()
        self.update_list()
        self.acc, self.virSum, self.uSum = calc_force(self.pos, self.neig_list)
        self.update_vel()
        
    def run(self, N_STEPS, N_OUT):
        print("NUM_PARTICLES =", str(NUM_PARTICLES), "\n")
        clock_start = time.perf_counter()
        for i in range(N_STEPS+1):
            self.single_step()
            if (i == 10000 or i%10000): 
                self.rescale_vel() 
            if (i % N_OUT) == 0: 
                self.calc_properties(i)
                self.append_trajectory()
        clock_end = time.perf_counter()
        print("\n"+"List updated " + str(self.list_updates) + " times")
        print("Simulation time " + str(clock_end-clock_start) + "s\n")
        return self.trajectory_pos, self.trajectory_vel, self.trajectory_acc

###########NUMBA FUNCTIONS#################
@numba.jit(nopython=True, parallel = False)
def build_neighbor_list(position):
    neighbor_list = np.full((NUM_PARTICLES -1, MAX_NEIGH+1),-1, numba.int32)
    # The last position neighbor_list(MAX_NEIGH+1) if for the list size
    for i in numba.prange(0, NUM_PARTICLES-1): 
        n=0
        for j in numba.prange(i+1, NUM_PARTICLES):
            dx = abs(position[i,0]-position[j,0])
            dy = abs(position[i,1]-position[j,1])
            dx=min(dx, LX - dx) 
            dy=min(dy, LY - dy)
            dr2 = (dx)**2+(dy)**2
            if dr2<CUT_OFF_LIST_2:
                neighbor_list[i,n] = j
                n +=1
                if n > MAX_NEIGH:
                    print("too many neighbors, increase MAX_NEIGH")
        neighbor_list[i,-1] = n # last element has the size of the list
    return neighbor_list

@numba.jit(nopython=True, parallel = False)# can't be parallel
def calc_force(position, neighbor_list):  
    force = np.zeros((NUM_PARTICLES, 2), dtype=numba.float64)
    virSum = 0.
    uSum = 0
    for i in numba.prange(0, NUM_PARTICLES - 1):
        n = neighbor_list[i,-1] # last element has the size of the i-list
        for k in numba.prange(0, n):
            j = neighbor_list[i,k]
            dx = position[i,0]-position[j,0]
            dy = position[i,1]-position[j,1]
            if abs(dx)>LX/2.: dx = (dx - np.sign(dx)*LX)
            if abs(dy)>LY/2.: dy = (dy - np.sign(dy)*LY)
            dr2 = numba.float64((dx)**2+(dy)**2)
            if dr2 < CUT_OFF_2:
                f = lj_f(dr2) #force/dr
                uSum += lj_u(dr2)
                virSum += f*dr2
                fx = f*dx
                fy = f*dy
                force[i,0] += fx
                force[i,1] += fy
                force[j,0] -= fx
                force[j,1] -= fy
    return force, virSum, uSum

@numba.vectorize([numba.float64(numba.float64)])
def lj_f(dr2):
    #input = dr**2
    #output = force/dr
    invr2 = 1./dr2
    invr6 = invr2*invr2*invr2
    f = 48.*invr6*(invr6-0.5)*invr2
    return f

@numba.vectorize([numba.float64(numba.float64)])
def lj_u(dr2):
    #input = distance square
    invr2 = 1./dr2
    invr6 = invr2*invr2*invr2
    U = 4.*invr6*(invr6-1.)
    return U

def make_video(trajectory):
    figure, axes = plt.subplots()
    camera = Camera(figure)
    axes.set_aspect(1)
    for pos in trajectory:
        plt.scatter(pos[:,0], pos[:,1], s=8000, marker='.', c='b')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('particle position')
        axes.set_xlim([0, LX])
        axes.set_ylim([0, LY])
        camera.snap()
    animation = camera.animate()
    animation.save('animation.mp4')

if __name__ == "__main__":
    simul = MD()
    trajectory_pos, trajectory_vel , trajectory_acc  = simul.run(N_STEPS, N_OUT)
    #make_video(trajectory_pos)

# Save the variables
np.save('trajectory_pos.npy', trajectory_pos)
np.save('trajectory_vel.npy', trajectory_vel)
np.save('trajectory_acc.npy', trajectory_acc)
