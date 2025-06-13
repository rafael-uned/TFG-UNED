import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from scipy import integrate
from scipy.integrate import simpson
from scipy import interpolate



def get_farfield_radiant_intensity_scalar(x, y, U, λ,  mu=1, eps=1):
    """
    Compute the far field radiance in a semisphere in α, β coordinates using scalar approx.

    Parameters
    ----------
    x, y : 1D arrays with spatial sampling arrays for the x and y coordinates.
    U : 2D Complex field distribution in the near field.
    λ : Wavelength of the light.
    output_region : list with the region [α0, αf, β0, βf] over which to evaluate the radiant intensity.
    mu, eps : magnetic permeability and electric permittivity (default 1).
    """
    Ny, Nx = U.shape
    # Compute the sampling intervals in x and y directions
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    #Use the natural FFT frequency grid
    fx = jnp.fft.fftshift(jnp.fft.fftfreq(Nx, d=dx)) # evaluates (Eq.1.2.2)
    fy = jnp.fft.fftshift(jnp.fft.fftfreq(Ny, d=dy))
    U_f = jnp.fft.fftshift(jnp.fft.fft2(U))
    
    fxx, fyy = jnp.meshgrid(fx, fy) #create 2d arrayswith the frequencies
    # Calculate a FT phase translation correction and scaling factor  (Eq.1.2.5)
    ft_factor = dx * dy * jnp.exp(-1j * 2 * jnp.pi * x[0] * fxx - 1j * 2 * jnp.pi * y[0] * fyy)
    U_f = U_f * ft_factor
    α = fx * λ # Convert frequency coordinates to angular coordinates α and β
    β = fy * λ
    αα, ββ = jnp.meshgrid(α, β)
    arg = 1 - αα**2 - ββ**2
    γγ = jnp.sqrt(jnp.where(arg > 0, arg, 0)) #discard evasnescent modes

    # Return the angular coordinates and the computed radiant_intensity distribution (Eq.1.1.31)
    radiant_intensity = 0.5 * jnp.sqrt(eps / mu) * jnp.real(U_f * jnp.conj(U_f)) * γγ / (λ**2)
    return α, β, radiant_intensity


#################################################################################################################################


def scalar_rayleigh_sommerfeld(x, y, U, x_eval, y_eval, z, λ):
    """
    Evaluates Eq. (1.1.18). The integral is computed using the trapezoid rule.
    
    Parameters
    ----------
    x, y: 1D arrays with x and y position samplings
    U: 2D array with the input plane field
    x_eval: 1D array giving the outplut x values
    y_eval: 1D array giving the outplut y values
    z: propagation distance
    λ: wavelength in the propagating medium
    """
    
    Ny,Nx = U.shape
    xx, yy = jnp.meshgrid(x, y)
    k = 2 * np.pi / λ
    U_eval = np.zeros(len(x_eval), dtype = 'complex64')

    for i in range(len(x_eval)):
        r = jnp.sqrt((x_eval[i]-xx)**2 + (y_eval[i]-yy)**2 + z**2)
        U_eval[i] = jax.scipy.integrate.trapezoid(jax.scipy.integrate.trapezoid(U / (1j*λ) * (1j/(k * r) + 1.)
                    * (jnp.exp(1j * k * r)/r * z/r ) , y, axis = 0), x, axis = 0)
    return U_eval


#################################################################################################################################


def scalar_ASM(x, y, U, z, λ):
    """
    Evaluates Eq. (1.1.10)
    
    Parameters
    ----------
    x, y: 1D arrays with x and y position samplings
    U: 2D array with the input plane field
    z: propagation distance
    λ: wavelength in the propagating medium
    """

    Ny, Nx = U.shape

    dx = x[1]-x[0]
    dy = y[1]-y[0]

    fx = jnp.fft.fftshift(jnp.fft.fftfreq(Nx, d = dx))  
    fy = jnp.fft.fftshift(jnp.fft.fftfreq(Ny, d = dy))   #evaluates Eq.(1.2.2)
    fxx, fyy = jnp.meshgrid(fx, fy)
    U_f = jnp.fft.fftshift(jnp.fft.fft2(U)) #evaluates Eq.(1.2.6)

    argument = (2 * jnp.pi)**2 * ((1. / λ) ** 2 - fxx ** 2 - fyy ** 2)
    #Calculate the propagating and the evanescent (complex) modes
    tmp = jnp.sqrt(jnp.abs(argument))
    kz = jnp.where(argument >= 0, tmp, 1j*tmp)
    U = jnp.fft.ifft2(jnp.fft.ifftshift(U_f * jnp.exp(1j * kz * z)))  #evaluates Eq.(1.1.9) and Eq.(1.2.8)
    return U


#################################################################################################################################


def get_modes_power_scalar(UG, λ, a1 = jnp.array([1.0, 0.0]), a2 = jnp.array([0.0, 1.0]), 
                           eps = 1,mu = 1):
    """
    Compute the power of each mode (m,n) per unit cell (Eq. 1.1.55)

    Parameters
    ----------
    UG: 2D array with the perodic field (Eq.1.1.40), in uv coordinates 
    λ: wavelength of the propagating medium
    a1, a2: primitive lattice vectors  (Eq.1.1.33)
    mu, eps : magnetic permeability and electric permittivity (default 1).
    """
    Nv,Nu = UG.shape
    J = jnp.abs(a1[0]*a2[1] - a1[1]*a2[0]) #Jacobian
    b1 = 2*jnp.pi / J * jnp.array([ a2[1], -a2[0] ])  #primitive reciprocal vectors (Eq. 1.1.36)
    b2 = 2*jnp.pi / J * jnp.array([-a1[1],  a1[0] ])
    m_, n_ = jnp.arange(Nu)-Nu//2 , jnp.arange(Nv)-Nv//2 # diffraction orders indexing
    mm_, nn_ = jnp.meshgrid(m_, n_)

    #Grating Equation
    Gmn_x, Gmn_y = mm_*b1[0] + nn_*b2[0], mm_*b1[1] + nn_*b2[1]
    senθx, senθy = 𝜆*Gmn_x/(2*np.pi),  𝜆*Gmn_y/(2*jnp.pi)
    #discard evanescent modes
    mask = (jnp.where( senθy**2 + senθx**2 >1, 0 , 1 ))
    Cmn = jnp.fft.fftshift(jnp.fft.fft2(UG))*(1/(Nv*Nu)) # (Eq. 1.1.42)
    pow_mode = 0.5* jnp.sqrt(eps/mu) * Cmn*jnp.conjugate(Cmn)* J * mask # (Eq. 1.1.55)
    return mm_, nn_, pow_mode


#################################################################################################################################


def get_farfield_radiant_intensity_percos_scalar(x, y, U, λ,  mu=1, eps=1):
    """
    Compute the far field radiance in a semisphere in α, β coordinates using scalar approx.

    Parameters
    ----------
    x, y : 1D arrays with spatial sampling arrays for the x and y coordinates.
    U : 2D Complex field distribution in the near field.
    λ : Wavelength of the light.
    output_region : list with the region [α0, αf, β0, βf] over which to evaluate the radiant intensity.
    mu, eps : magnetic permeability and electric permittivity (default 1).
    """
    Ny, Nx = U.shape
    # Compute the sampling intervals in x and y directions
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    #Use the natural FFT frequency grid
    fx = jnp.fft.fftshift(jnp.fft.fftfreq(Nx, d=dx)) # evaluates (Eq.1.2.2)
    fy = jnp.fft.fftshift(jnp.fft.fftfreq(Ny, d=dy))
    U_f = jnp.fft.fftshift(jnp.fft.fft2(U))
    
    fxx, fyy = jnp.meshgrid(fx, fy) #create 2d arrayswith the frequencies
    # Calculate a FT phase translation correction and scaling factor  (Eq.1.2.5)
    ft_factor = dx * dy * jnp.exp(-1j * 2 * jnp.pi * x[0] * fxx - 1j * 2 * jnp.pi * y[0] * fyy)
    U_f = U_f * ft_factor
    α = fx * λ # Convert frequency coordinates to angular coordinates α and β
    β = fy * λ
    αα, ββ = jnp.meshgrid(α, β)
    arg = 1 - αα**2 - ββ**2
    γγ = jnp.sqrt(jnp.where(arg > 0, arg, 0)) #discard evasnescent modes

    # Return the angular coordinates and the computed radiant_intensity distribution (Eq.1.1.31)
    radiant_intensity = 0.5 * jnp.sqrt(eps / mu) * jnp.real(U_f * jnp.conj(U_f)) / (λ**2)
    return α, β, radiant_intensity


#################################################################################################################################


def get_MA_nearfield_phase_fun(input_fun, target_fun , extent_input, extent_target, λ, z, integration_points = 500000):
    """
    Solves Eq.(2.1.11) using the method described in section 2.1.4
    
    Parameters
    ----------
    input_fun, target_fun: 1D functions with I(r) and E(t) as defined in section 2.1.2 
    extent_input, extent_target: float values with 
    λ, z: wavelength in the propagating medium and target profile distance
    integration_points: number of samples to use for the numerical integration
    """
    r = np.linspace(0,extent_input, integration_points) #input r coordinates
    I = input_fun(r) #incident intensity profile
    t = np.linspace(0,extent_target, integration_points) #target t coordinates
    E = target_fun(t) #target intensity profile
    PI,PE = simpson(r*I, r), simpson(t*E, t) # compute total power
    E = PI/PE*E  # scale the target profile to conserve energy
    int_I = np.array(integrate.cumtrapz(r*I, r, initial=0) ) #evaluate left hand side of Eq.(2.1.19)
    int_E = np.array(integrate.cumtrapz(t*E, t, initial=0) ) #evaluate right hand side of Eq.(2.1.19)
    int_E, idx = np.unique(int_E, return_index=True) # remove repeated values
    t = t[idx]
    
    int_E_inv_fun = interpolate.interp1d(int_E, t, kind="linear",bounds_error=False ,fill_value = (t.min(), t.max()))
    t = int_E_inv_fun(int_I) #evaluate Eq.(2.1.20)
    T_fun = interpolate.interp1d(r, t, kind="cubic",bounds_error=False, fill_value = (t.min(), t.max()))
    t =  T_fun(r) #define T(R) from Eq.(2.1.20)  using interpolation
    
    dΦ_dr = (t - r) / np.sqrt(z**2 +  (t - r)**2 ) # Right hand side of Eq.(2.1.21)
    Φ = (2*np.pi / λ)  * integrate.cumtrapz(dΦ_dr, r, initial=0)  #Integrate Eq.(2.1.21)
    Φ_fun = interpolate.interp1d(r, Φ - Φ[0], kind="cubic",bounds_error=False , fill_value = Φ.max())
    return Φ_fun, PI/ PE


#################################################################################################################################


def get_MA_farfield_phase_fun(input_fun, target_fun, extent_r, λ, integration_points = 500000):
    """
    Solves Eq.(2.1.17) using the method described in section 2.1.4
    
    Parameters
    ----------
    input_fun, target_fun: 1D functions with I(r) and IeΩcosθ(θ) as defined in section 2.1.2 
    extent_input, extent_target: float values with 
    λ, z: wavelength in the propagating medium and target profile distance
    integration_points: number of samples to use for the numerical integration
    """
    r = np.linspace(0,extent_r, integration_points) #input r coordinates
    I = input_fun(r) # incident instensity profile
    θ = np.linspace(0,np.pi/2, integration_points) #target θ coordinates
    IeΩcosθ = target_fun(np.sin(θ)) #target intensity profile I_e,Ωcosθ(α,β)
    PI,PT = simpson(r*I, r),   simpson(np.sin(θ) * np.cos(θ)*IeΩcosθ, θ) # compute total power
    IeΩcosθ = PI/PT*IeΩcosθ # scale the target profile to conserve energy
    int_I = np.array(integrate.cumtrapz(r*I, r, initial=0) )#evaluate LHS of Eq.(2.1.22)
    int_IeΩcosθ =np.array(integrate.cumtrapz(np.sin(θ)*np.cos(θ)*IeΩcosθ, θ, initial=0))#evaluate RHS of Eq.(2.1.22)
    int_IeΩcosθ, idx = np.unique(int_IeΩcosθ, return_index=True) # remove repeated values
    θ = θ[idx]
    int_IeΩcosθinv_interpolate = interpolate.interp1d(int_IeΩcosθ,θ,kind="linear", bounds_error=False, 
                                                                      fill_value = (θ.min(), θ.max()))
    θ = int_IeΩcosθinv_interpolate(int_I) #evaluate Eq.(2.1.23)
    Θ_fun = interpolate.interp1d(r, θ, kind="cubic",bounds_error=False, fill_value = (θ.min(), θ.max()))
    θ =  Θ_fun(r) #define Θ(R) from Eq.(2.1.23)  using interpolation    
    dΦ_dr = np.sin(θ) # Right hand side of  Eq.(2.1.24)
    Φ = (2*np.pi / λ)  * integrate.cumtrapz(dΦ_dr, r, initial=0)  #Integrate Eq.(2.1.24)
    Φ_fun = interpolate.interp1d(r, Φ-Φ[0], kind="cubic",bounds_error=False , fill_value = Φ.max())
    return Φ_fun, PI/ PT


#################################################################################################################################


def GS_phase_retrieval(input_fun,target_fun, extent_x, extent_y, Nx, Ny,num_iter = 40):
    """
    Calculates the phase profile using the Gerchberg-Saxton algorithm described in section 2.1.6
    
    Parameters
    ----------
    input_fun, target_fun: 2D functions with input and target functions
    extent_x, extent_y: float values with extent of the input function
    Nx, Ny: Number of points in which we divide each axis extent
    num_iter: number of iterations
    """
    dx, dy = extent_x/Nx,  extent_y/Ny
    x, y = dx*(jnp.arange(Nx)-Nx//2),  dy*(jnp.arange(Ny)-Ny//2)
    xx, yy = jnp.meshgrid(x, y)
    fx, fy = jnp.fft.fftshift(jnp.fft.fftfreq(Nx, d = dx)) , jnp.fft.fftshift(jnp.fft.fftfreq(Ny, d = dy))
    fxx, fyy = jnp.meshgrid(fx, fy)
    source_amplitude = jnp.abs(jnp.fft.ifftshift(jnp.sqrt(input_fun(xx, yy))))
    target_amplitude = jnp.abs(jnp.fft.ifftshift(jnp.sqrt(target_fun(fxx, fyy))/(dx*dy)))
    error_list = []
    U_p = jnp.fft.ifft2(jnp.fft.ifftshift(target_amplitude)) # Step 1  (from section 2.1.6)
    for iter in range(num_iter):
        U = source_amplitude * jnp.exp(1j * jnp.angle(U_p)) # Step 2
        Uf = jnp.fft.fft2(U) # Step 3
        Uf_p = target_amplitude * jnp.exp(1j * jnp.angle(Uf)) # Step 4
        U_p = jnp.fft.ifft2(Uf_p)  # Step 5
        
        diff = jnp.abs(Uf)  - target_amplitude
        squared_err = (jnp.mean(diff**2))
        error_list += [squared_err]
    return jnp.fft.fftshift(jnp.angle(U_p)) , error_list






