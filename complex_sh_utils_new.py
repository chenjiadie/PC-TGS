import numpy as np
import torch
import pdb
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

##Reference:
# https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Complex_spherical_harmonics
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5068319
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5594365
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1284991

C0 = 0.28209479177387814

C1 = [
    0.3454941494713355,
    0.4886025119029199,
    -0.3454941494713355
]
C2 = [
    0.3862742020231896,
    0.7725484040463791,
    0.31539156525252005,
    -0.7725484040463791,
    0.3862742020231896
]
C3 = [
    0.4172238236327841,
    1.0219854764332823,
    0.32318018411415067,
    0.3731763325901154,
    -0.32318018411415067,
    1.0219854764332823,
    -0.4172238236327841,
]

C4 = [
    0.4425326924449826,
    1.2516714708983523,
    0.3345232717786446,
    0.47308734787878004,
    0.10578554691520431,
    -0.47308734787878004,
    0.3345232717786446,
    -1.2516714708983523,
    0.4425326924449826
]



def eval_sh(deg, sh_real,sh_imag, dirs1,dirs2):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh_real: jnp.ndarray SH coeffs real part [..., C, (deg + 1) ** 2]
        sh_imag: jnp.ndarray SH coeffs imaginary part [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh_real.shape[-1] >= coeff
    assert sh_imag.shape[-1] >= coeff
    sh=sh_real+1j*sh_imag

    result=(sh[..., 0]*C0*(1+1j*0)*C0*(1+1j*0)).unsqueeze(0)

    if deg > 0:
        x1, y1, z1 = dirs1[..., 0:1], dirs1[..., 1:2], dirs1[..., 2:3]
        x2, y2, z2 = dirs2[..., 0:1], dirs2[..., 1:2], dirs2[..., 2:3]
        result=(result+
                C1[0]*C1[0]*sh[..., 1]*(x1-1j*y1)*(x2-1j*y2)+
                C1[1]*C1[1]*sh[..., 2]*(z1+1j*0)*(z2+1j*0)+
                C1[2]*C1[2]*sh[..., 3]*(x1+1j*y1)*(x2+1j*y2))

        if deg > 1:
            result=(result+
                C2[0]*C2[0]*sh[..., 4]*(x1-1j*y1)*(x1-1j*y1)*(x2-1j*y2)*(x2-1j*y2)+
                C2[1]*C2[1]*sh[..., 5]*(x1-1j*y1)*z1*(x2-1j*y2)*z2+
                C2[2]*C2[2]*sh[..., 6]*(3*z1**2-1+1j*0)*(3*z2**2-1+1j*0)+
                C2[3]*C2[3]*sh[..., 7]*(x1+1j*y1)*z1*(x2+1j*y2)*z2+
                C2[4]*C2[4]*sh[..., 8]*(x1+1j*y1)*(x1+1j*y1)*(x2+1j*y2)*(x2+1j*y2)
                )

            if deg > 2:
                result=(result+
                C3[0]*C3[0]*sh[..., 9]*(x1-1j*y1)*(x1-1j*y1)*(x1-1j*y1)*(x2-1j*y2)*(x2-1j*y2)*(x2-1j*y2)+
                C3[1]*C3[1]*sh[..., 10]*(x1-1j*y1)*(x1-1j*y1)*z1*(x2-1j*y2)*(x2-1j*y2)*z2+
                C3[2]*C3[2]*sh[..., 11]*(x1-1j*y1)*(5*z1**2-1)*(x2-1j*y2)*(5*z2**2-1)+
                C3[3]*C3[3]*sh[..., 12]*(5*z1**3-3*z1+1j*0)*(5*z2**3-3*z2+1j*0)+
                C3[4]*C3[4]*sh[..., 13]*(x1+1j*y1)*(5*z1**2-1)*(x2+1j*y2)*(5*z2**2-1)+
                C3[5]*C3[5]*sh[..., 14]*(x1+1j*y1)*(x1+1j*y1)*z1*(x2+1j*y2)*(x2+1j*y2)*z2+
                C3[6]*C3[6]*sh[..., 15]*(x1+1j*y1)*(x1+1j*y1)*(x1+1j*y1)*(x2+1j*y2)*(x2+1j*y2)*(x2+1j*y2)
                )

                if deg > 3:
                    result=(result+
                    C4[0]*C4[0]*sh[..., 16]*(x1-1j*y1)*(x1-1j*y1)*(x1-1j*y1)*(x1-1j*y1)*(x2-1j*y2)*(x2-1j*y2)*(x2-1j*y2)*(x2-1j*y2)+
                    C4[1]*C4[1]*sh[..., 17]*(x1-1j*y1)*(x1-1j*y1)*(x1-1j*y1)*z1*(x2-1j*y2)*(x2-1j*y2)*(x2-1j*y2)*z2+
                    C4[2]*C4[2]*sh[..., 18]*(x1-1j*y1)*(x1-1j*y1)*(7*z1**2-1)*(x2-1j*y2)*(x2-1j*y2)*(7*z2**2-1)+
                    C4[3]*C4[3]*sh[..., 19]*(x1-1j*y1)*(7*z1**3-3*z1)*(x2-1j*y2)*(7*z2**3-3*z2)+
                    C4[4]*C4[4]*sh[..., 20]*(35*z1**4-30*z1**2+3+1j*0)*(35*z2**4-30*z2**2+3+1j*0)+
                    C4[5]*C4[5]*sh[..., 21]*(x1+1j*y1)*(7*z1**3-3*z1)*(x2+1j*y2)*(7*z2**3-3*z2)+
                    C4[6]*C4[6]*sh[..., 22]*(x1+1j*y1)*(x1+1j*y1)*(7*z1**2-1+1j*0)*(x2+1j*y2)*(x2+1j*y2)*(7*z2**2-1+1j*0)+
                    C4[7]*C4[7]*sh[..., 23]*(x1+1j*y1)*(x1+1j*y1)*(x1+1j*y1)*z1*(x2+1j*y2)*(x2+1j*y2)*(x2+1j*y2)*z2+
                    C4[8]*C4[8]*sh[..., 24]*(x1+1j*y1)*(x1+1j*y1)*(x1+1j*y1)*(x1+1j*y1)*(x2+1j*y2)*(x2+1j*y2)*(x2+1j*y2)*(x2+1j*y2)
                    )

    return result




if __name__ == "__main__":
    deg = 4
    N = 100
    device='cuda:0'

    sh_real = torch.randn((N,1,(deg+1)**2), requires_grad=True,device=device)
    sh_imag = torch.randn((N,1,(deg+1)**2), requires_grad=True,device=device)
    dirs = torch.randn((N, 3),device=device)

    print("sh_real shape:", sh_real.shape)
    print("sh_real mean:", sh_real.abs().mean().item())
    print("sh_imag shape:", sh_imag.shape)
    print("sh_imag mean:", sh_imag.abs().mean().item())

    complex_s = eval_sh(deg, sh_real, sh_imag, dirs)

    print('==============')

    print('complex_s shape:',complex_s.shape)

    print('==============')

    loss = torch.norm(complex_s, p=2)
    loss.backward()

    print("sh_real.grad shape:", sh_real.grad.shape)
    print("sh_real.grad mean:", sh_real.grad.abs().mean().item())
    print("sh_imag.grad shape:", sh_imag.grad.shape)
    print("sh_imag.grad mean:", sh_imag.grad.abs().mean().item())
