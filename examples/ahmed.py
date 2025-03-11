import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    EquilibriumBC,
    DoNothingBC,
    RegularizedBC,
    HalfwayBounceBackBC,
    ExtrapolationOutflowBC,
    HybridBC,
)
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image, save_BCs_vtk, q_criterion
from xlb.helper import initialize_eq
import trimesh, time, os, sys
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt



# -------------------------- User Setup --------------------------
def main():
    # clear kernel cash
    wp.clear_kernel_cache()
    # stl to load
    stl_name = 'stls/Ahmed_25.stl'
    # Ref area in m2 
    ref_area = 0.112
    cd_exp = 0.28
    cl_exp = -0.014
    # Output folder name
    output_name = 'Ahmed_FBB'

    # Domain size multiple of length
    #grid_multiplier = [2.0 , 2.0 , 1.0]    # MEDIUM DOMAIN
    grid_multiplier = [2.0 , 1.0 , 0.6]     # SMALL DOMAIN


    # Assume meters
    voxel_size = 0.004

    # Assume meters / second (current Exp Data is 40m/s )
    velocity = 40.0
    ulb = 0.05

    # Flow Passes
    flow_passes = 5


    # Cutover from low i/o to averaging segment
    cutover = 0.8
    initial_saved_frames = 4
    averaging_frames = 2
    averaging_forces = 20
    
    solve(stl_name, ref_area, output_name, grid_multiplier, voxel_size, velocity, ulb, flow_passes, cutover, initial_saved_frames, averaging_frames, averaging_forces, cd_exp, cl_exp)
    



# -------------------------- Simulation Setup --------------------------
def solve(stl_name, ref_area, output_name, grid_multiplier, voxel_size, velocity, ulb, flow_passes, cutover, initial_saved_frames, averaging_frames, averaging_forces, cd_exp, cl_exp):
    

    # Grid parameters
    # ---------------
    current_dir = os.path.join(os.path.dirname(__file__))
    drag_stl = os.path.join(current_dir, stl_name)
    drag_mesh = trimesh.load_mesh(drag_stl, process=False)
    drag_length = drag_mesh.extents[0]
    # Output Directory
    output_dir = os.path.join(current_dir, output_name)    

    # Setup grid shape
    # ---------------
    grid_size_x = int(grid_multiplier[0] * drag_length / voxel_size)
    grid_size_y = int(grid_multiplier[1] * drag_length / voxel_size) 
    grid_size_z = int(grid_multiplier[2] * drag_length / voxel_size)  
    grid_shape = (grid_size_x, grid_size_y, grid_size_z)

    # Fluid Properties
    air_kin_visc = 1.508e-5
    
    
    
    dt = get_physical_timestep(voxel_size, velocity, ulb)
    visc_lbm = air_kin_visc * (dt / voxel_size**2)
    omega = 1.0 / (3. * visc_lbm + 0.5)
    
 
    # Clean out old results if they exist
    if os.path.exists(output_dir):
       os.system("rm -r "+ output_dir)
    # Start new folder for results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_steps = int(flow_passes*(grid_shape[0]/ulb))
    # Configuration
    compute_backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)



    # Print simulation info
    print("\n" + "=" * 50 + "\n")
    print("Simulation Configuration:")
    print(f"Grid size: {grid_size_x} x {grid_size_y} x {grid_size_z}")
    print(f"Voxel Count: {grid_size_x*grid_size_y*grid_size_z:,}")
    print(f"compute_backend: {compute_backend}")
    print(f"Velocity set: {velocity_set}")
    print(f"Precision policy: {precision_policy}")
    print(f"Prescribed velocity: {ulb}")
    print(f"Max iterations: {num_steps:,}")
    print("\n" + "=" * 50 + "\n")
    
    
    # Initialize XLB
    # ---------------
    wp.config.max_unroll = 27
    xlb.init(
        velocity_set=velocity_set,
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )

    # Create Grid
    grid = grid_factory(grid_shape, compute_backend=compute_backend)

    # ---------------- Setup Indices ------------------
    # -----------------------------------------------------------------------    
    
    boundingBoxIndices = grid.bounding_box_indices()
    boundingBoxIndices_noEdge = grid.bounding_box_indices(remove_edges=True)
    
    inlet = boundingBoxIndices_noEdge['left']
    outlet = boundingBoxIndices_noEdge["right"]
    #self.outlet = outlet
    
    
    walls = [
        boundingBoxIndices["back"][i]
        + boundingBoxIndices["top"][i]
        + boundingBoxIndices["front"][i]            
        for i in range(velocity_set.d)
     ]
    walls = np.unique(np.array(walls), axis=-1).tolist()
     
    ground = [
        boundingBoxIndices["bottom"][i]
        for i in range(velocity_set.d)
     ]

    #Remove overlap between walls and ground
    ground_x ,ground_y ,ground_z  = np.array(ground[0]), np.array(ground[1]), np.array(ground[2]) 
    ground_y_min = np.min(ground_y)
    ground_y_max = np.max(ground_y)
    ground_mask_ymin = ground_y == ground_y_min
    ground_mask_ymax = ground_y == ground_y_max
    ground_mask = ground_mask_ymax | ground_mask_ymin
    ground_filter = np.where(~ground_mask)[0]
    
    ground = [
        tuple(ground_x[ground_filter]),
        tuple(ground_y[ground_filter]),
        tuple(ground_z[ground_filter])
        ]

    mesh_vertices = drag_mesh.vertices
    
    # Get the minimum vertex coordinates along each axis (x, y, z)
    min_values = mesh_vertices.min(axis=0)        

    # Transform the mesh points to be located in the right position in the wind tunnel
    mesh_vertices -= mesh_vertices.min(axis=0)
    mesh_extents = mesh_vertices.max(axis=0)
    mesh_vertices = mesh_vertices / voxel_size
    # Shift car to domain location
    shift_X = (grid_shape[0]/7) 
    shift_Y = (0.5 * (grid_shape[1] - mesh_extents[1] / voxel_size))
    shift_Z = (0.0)         
    shift = np.array([shift_X, shift_Y, shift_Z]).astype(int)         
    car = mesh_vertices + shift
    
    # Setup Origin  
    org_X = min_values[0] - (shift[0])*voxel_size    
    org_Y = min_values[1] - (shift[1])*voxel_size
    org_Z = min_values[2] - (shift[2])*voxel_size
    origin = np.array([org_X, org_Y, org_Z])        
    
    
    drag_rear = ((drag_mesh.bounds[1] - drag_mesh.bounds[0])/voxel_size) + shift 
    # x0 is center of wheelbase
    x0 = int(drag_rear[0])-int(0.603/voxel_size)
    
    ref_area = ref_area /voxel_size**2
    
    # -----------------     Setup Boundary Conditions   ---------------------
    # -----------------------------------------------------------------------
    bc_inlet = RegularizedBC('velocity', prescribed_value=(ulb, 0.0, 0.0), indices=inlet)
    bc_outlet = DoNothingBC(indices=outlet)    
    bc_walls = FullwayBounceBackBC(indices=walls)
    #bc_walls = HybridBC(bc_method="nonequilibrium_regularized", indices=walls, prescribed_value=(0.0, 0.0, 0.0))
    #bc_walls = HybridBC(bc_method="nonequilibrium_regularized", indices=walls)
    bc_ground = FullwayBounceBackBC(indices=ground)
    #bc_ground = HybridBC(bc_method="nonequilibrium_regularized", indices=ground, prescribed_value=(0.0, 0.0, 0.0))
    #bc_ground = HybridBC(bc_method="nonequilibrium_regularized", indices=ground)
    
    bc_car = FullwayBounceBackBC(mesh_vertices=car, voxelization_method="aabb")
    #bc_car = HybridBC(bc_method="nonequilibrium_regularized", mesh_vertices=car, use_mesh_distance=True, voxelization_method="aabb")
    boundary_conditions = [bc_inlet, bc_outlet, bc_walls, bc_ground, bc_car]
    
    
    # -------------------     Setup Stepper    -----------------
    # -----------------------------------------------------------------------
    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="KBC",
    ) 
    
    # Prepare Fields
    # ---------------
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()
    
    # Initialize Fields
    # ---------------
    shape = (velocity_set.d,) + grid_shape
    outlet_edge = np.array(outlet)
    outlet_edge = outlet_edge.T

    u_init = np.zeros(shape)
    x, y, z, = outlet_edge[:,0], outlet_edge[:,1], outlet_edge[:,2]
    u_init[0,x,y,z] = ulb
        
    if compute_backend == ComputeBackend.JAX:
        u_init = jnp.full(shape=shape, fill_value=u_init)
    else:
        u_init = wp.array(u_init , dtype=precision_policy.compute_precision.wp_dtype)
    f_0 = initialize_eq(f_0, grid, velocity_set, precision_policy, compute_backend, u=u_init)


    # Setup Momentum Transfer for Force Calculation
    bc_car = boundary_conditions[-1]
    momentum_transfer = MomentumTransfer(bc_car, compute_backend=compute_backend)

    # Define Macroscopic Calculation
    macro = Macroscopic(
        compute_backend=ComputeBackend.JAX,
        precision_policy=precision_policy,
        velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
    )

    # Initialize Lists to Store Coefficients and Time Steps
    dataout={}
    dataout['count']= 0
    dataout['step'] = []
    dataout['cd']=[]
    dataout['cl']=[]
    dataout['cy']=[]
    dataout['u'] = 0
    
    

    # -------------------------- Simulation Loop --------------------------
    # ---------------------------------------------------------------------
    
    # Prepare saving  with limited i/o
    first_run = int(num_steps * cutover)
    first_save = int(first_run/initial_saved_frames)
    second_run = (num_steps-first_run)
    second_save = int(second_run/averaging_frames)
    save_drag = int(second_run/averaging_forces)
    
    # Calculate the interval for 1% of the total iterations
    performance_output_intervals = 100
    perf_iterations_per_update = int(max(1, num_steps // 100))
    total_lattice_points = grid_shape[0] * grid_shape[1] * grid_shape[2]

    start_time = time.time()
    perf_time = time.time()
    for step in range(num_steps):
        if step % perf_iterations_per_update == 0:
            interval_time = time.time() - perf_time
            mlups = (total_lattice_points * perf_iterations_per_update) / (interval_time * 1e6)  # Million Lattice Updates per Second
            perf_time = time.time()
            print(f"Progress: Iteration {step}/{num_steps}, MLUPs: {mlups:.2f}")
        
        # Perform simulation step
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
        #f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, step)
        f_0, f_1 = f_1, f_0  # Swap the buffers



        # Print progress at intervals
        if step < first_run:
            if step % first_save == 0:                            
                if compute_backend == ComputeBackend.WARP:
                    
                    elapsed_time = time.time() - start_time
                    print("")
                    print(f"----Iteration: {step}/{num_steps} ----")
                    print(f"Flow Passes : {step*ulb/grid_shape[0]}")
                    print(f"Time Elapsed: {elapsed_time:.2f}s")
                    rho, u = output_data(
                        step,
                        f_0,
                        macro,
                        bc_mask,
                        voxel_size,
                        dt,
                        omega,
                        output_dir,
                        origin,
                        )
                    cd, cl, cy = compute_drag(
                        momentum_transfer,
                        f_0,
                        f_1,
                        bc_mask,
                        missing_mask,
                        ref_area,
                        ulb,
                    )    
                    print(f"Cd : {cd} | Cl : {cl}")
                    
            
           
        # Beyond the cutover
        else:
            # Post-process Results or Drag at intervals and final step
            if (step % save_drag == 0) or (step == num_steps - 1) or (step % second_save == 0):
                elapsed_time = time.time() - start_time
                print("")
                print(f"----Iteration: {step}/{num_steps} ----")
                print(f"Flow Passes : {step*ulb/grid_shape[0]}")
                print(f"Time Elapsed: {elapsed_time:.2f}s")
                if step % save_drag == 0:
                    cd, cl, cy = compute_drag(
                            momentum_transfer,
                            f_0,
                            f_1,
                            bc_mask,
                            missing_mask,
                            ref_area,
                            ulb
                    )    
                    print(f"Cd : {cd} | Cl : {cl}")
                    dataout['cd'].append(cd)
                    dataout['cl'].append(cl)
                    dataout['cy'].append(cy)
                    dataout['step'].append(step)
                
                else:
                    rho, u = output_data(
                            step,
                            f_0,
                            macro,
                            bc_mask,
                            voxel_size,
                            dt,
                            omega,
                            output_dir,
                            origin,
                    )
                    dataout['count']= dataout['count'] +1
                    dataout['u'] = dataout['u'] + u
                    
                
    wp.synchronize()            
    plot_drag(
        cd_exp,
        cl_exp,
        dataout,    
        output_dir,
        prefix='drivear_estate',
    )            
                

    print("Simulation completed successfully.")

# -------------------------- Helper Functions --------------------------
def get_physical_timestep(physical_discretization_step, physical_velocity, ulb):
    dx = physical_discretization_step       # meters
    dt = dx * ulb / abs(physical_velocity)
    return dt 

def output_data(
    step,
    f_0,
    macro,
    bc_mask,
    voxel_size,
    dt,
    omega,
    output_dir,
    origin
    ):
    wp.synchronize()
    start_time = time.time()    
    # Convert to JAX array if necessary
    if not isinstance(f_0, jnp.ndarray):
        f_0_jax = wp.to_jax(f_0)
        if step == 0:
                bc_mask = wp.to_jax(bc_mask)[0]
    else:
        f_0_jax = f_0

    # Compute macroscopic quantities
    rho, u = macro(f_0_jax)
    u = u[:, :, :, :] * voxel_size / dt       
    rho = rho[:, :, :, :] 
    # Output vtk and image
    #mu, q, tau_xy, tau_xz, tau_yz, tau_magnitude = q_criterion(u, omega)
    # u = u[:, 1:-1, 1:-1, 1:-1]
    # rho = rho[:, 1:-1, 1:-1, 1:-1]
    print(f"Time to compute Macro Total: {time.time()-start_time} sec")
    # fields = {"umag": (u[0]**2+u[1]**2+u[2]**2)**0.5,
              # "rho" : rho[0],
              # "tau_xy"  : tau_xy,
              # "tau_xz"  : tau_xz,
              # "tau_yz"  : tau_yz,
              # "tau_magnitude"  : tau_magnitude,
              # "mu"  : mu,
              # "q"   : q}
    fields = {"umag": (u[0]**2+u[1]**2+u[2]**2)**0.5, 
              "rho" : rho[0],
              #"tau_magnitude"  : tau_magnitude
}
    
    save_fields_vtk(fields, timestep=step, output_dir=output_dir,shift_coords=(origin[0], origin[1], origin[2]),scale=voxel_size)
    
    if step == 0:    
        save_fields_vtk({"BCs": bc_mask}, timestep=step, prefix="BC_field", output_dir=output_dir,shift_coords=(origin[0], origin[1], origin[2]),scale=voxel_size)
        
    if np.isnan(np.average(u[0])):
        print("NaN in Velocity")
        sys.exit()
    return rho, u

def compute_drag(
    momentum_transfer,
    f_0,
    f_1,
    bc_mask,
    missing_mask,
    ref_area,
    ulb,
    ):
    wp.synchronize()
    start_time = time.time()
    # Compute lift and drag
    boundary_force = momentum_transfer(f_0, f_1, bc_mask, missing_mask)
    # Setup Reference Area
    ref_area
    drag = boundary_force[0]
    yaw = boundary_force[1]
    lift = boundary_force[2]
    c_d = 2.0 * drag / (ulb**2 * ref_area)
    c_y = 2.0 * yaw  / (ulb**2 * ref_area)
    c_l = 2.0 * lift / (ulb**2 * ref_area)
    print(f"Time to compute Drag Total: {time.time()-start_time} sec")
    
    return c_d, c_l, c_y

def plot_drag(
    cd_exp,
    cl_exp,
    dataout,    
    output_dir,
    prefix='Body',
    ):
    #Plot Cd comparison
    drag_data = [cd_exp, np.average(dataout['cd'])]
    lift_data = [cl_exp, np.average(dataout['cl'])]
    x1 = ["Experimental Cd", "XLB Cd"]
    x2 = ["Experimental Cl", "XLB Cl"]
    plt.bar(x1,drag_data,color=['black', 'blue'])
    # Add values on top of each bar
    for i, val in enumerate(drag_data):
        plt.text(i, val - 0.1, f'{val:.3f}', ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
    plt.ylabel("Drag Coefficient")
    plt.savefig(f"{output_dir }/{prefix}_Drag_Coeff.png")
    plt.close()
    
    plt.bar(x2,lift_data,color=['black', 'blue'])
    # Add values on top of each bar
    for i, val in enumerate(lift_data):
        plt.text(i, val - 0.1, f'{val:.3f}', ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
    plt.ylabel("Lift Coefficient")
    plt.savefig(f"{output_dir }/{prefix}_Lift_Coeff.png")
    plt.close()
    
    #Plot Cd vs iteration
    plt.plot(dataout['step'],dataout['cd'], "kx", label="Cd")     
    plt.legend(loc='upper left')
    plt.ylim(-1.2,1.2)
    plt.savefig(f"{output_dir}/Drag_v_Iteration.png")
    plt.close()
    #Plot Cl vs iteration        
    plt.plot(dataout['step'],dataout['cl'], "bx", label="Cl")      
    plt.legend(loc='upper left')        
    plt.savefig(f"{output_dir}/Lift_v_Iteration.png")
    plt.close()
    #Plot Cy vs iteration      
    plt.plot(dataout['step'],dataout['cy'], "gx", label="Cy")      
    plt.legend(loc='upper left')        
    plt.savefig(f"{output_dir}/Yaw_v_Iteration.png")
    plt.close()
    label = prefix + "_Coefficients.csv"
    with open(os.path.join(output_dir, label), 'w') as fd:
            csvHeader = "Iteration, Cd, Cl, Cy,\n"
            fd.write(csvHeader)                
            for n in range(len(dataout['step'])):
                csvRow = f"{dataout['step'][n]}, {dataout['cd'][n]}, {dataout['cl'][n]}, {dataout['cy'][n]}\n"
                fd.write(csvRow)



if __name__ == "__main__":
    main()