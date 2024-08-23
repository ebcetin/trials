import MDAnalysis as mda
import numpy as np
from MDAnalysis.tests.datafiles import PSF, DCD
from MDAnalysis.analysis import distances
import matplotlib.pyplot as plt



u = mda.Universe('new33.psf', 'new33.dcd')

hydride=u.select_atoms('resname QM and name H2')
print(hydride)
donor=u.select_atoms('resname QM and name C2')
acceptor=u.select_atoms('resname QM and name C4N')

d=np.zeros((501,1,1))
a=np.zeros((501,1,3))
b=np.zeros((501,1,3))
h=np.zeros((501,1,3))
count=0
for ts in u.trajectory:
	a[count]=donor.ts.positions
	b[count]=acceptor.ts.positions
	h[count]=hydride.ts.positions
	count=count+1

position_corr=np.zeros((501,1))
for i in range(501):
	d[i]=np.dot((h[i]-a[i])/np.linalg.norm(h[i]-a[i]),np.transpose((b[i]-h[i])/np.linalg.norm(b[i]-h[i])))
	position_corr[i]=d[i][0][0]

print(min(position_corr))
plt.plot(np.linspace(0,501,501),position_corr)

plt.show()

def load_velocities_from_binary(velocity_file, num_frames, num_particles, dtype='float32'):
    """
    Load velocities from a binary file.
    
    Parameters:
    - velocity_file: str, path to the binary velocity file
    - num_frames: int, number of frames in the simulation
    - num_particles: int, number of particles in the simulation
    - dtype: str, data type of the binary file (default is 'float32')
    
    Returns:
    - velocities: numpy array of shape (num_frames, num_particles, 3), velocities for all particles over time
    """
    # Calculate the total number of elements (frames * particles * 3 coordinates)
    total_elements = num_frames * num_particles * 3
    
    # Load the binary data into a 1D numpy array
    raw_data = np.fromfile(velocity_file, dtype=dtype, count=total_elements)
    
    # Reshape the 1D array into the 3D shape (num_frames, num_particles, 3)
    velocities = raw_data.reshape((num_frames, num_particles, 3))
    
    return velocities

velocities=load_velocities_from_binary('new33.vel',500,126113,dtype='float32')

#print(velocities.shape)
def momentum(a,timestep,resid_id):
	#mass = a.masses
	mass=1
	velocity = velocities[timestep,resid_id,:]
	velocity = velocity/np.linalg.norm(velocity)
	momenta = mass*velocity
	return momenta

momentum_e=np.zeros((500,347,3))  # an array of time * residues are tabulated
momentum_b=np.zeros((500,1,3))
momentum_c=np.zeros((500,1,3))

#print(momentum(acceptor,0,1))
for i in range(1,346):
	e=u.select_atoms(f'(segid PROA and resid {i} and name CA)')
	atom_id=e.ids 
	for j in range(500):
		momentum_e[j,i,:] = momentum(e,j,atom_id) # u_Dc
		momentum_b[j,0,:] = momentum(acceptor,j,20803) # u_bc
		momentum_c[j,0,:] = momentum(donor,j,20815) # u_ac

i=0

corr=np.zeros((500,500))
nnz=np.zeros((500,1))
f_corr=np.zeros((500,1))
for t in range(0,500):
	for j in range(t-j):
		corr[t,j] += np.dot(momentum_e[t,i,:],momentum_e[j+t-1,i,:])
		f_corr[t]=corr/nnz[j,:]

plt.plot(corr)
plt.show()





