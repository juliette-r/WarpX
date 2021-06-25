/* Copyright 2019 Andrew Myers, Aurore Blelly, Axel Huebl
 * David Grote, Maxence Thevenet, Remi Lehe
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "WarpX.H"
#include "Utils/WarpXConst.H"
#include "BoundaryConditions/WarpX_PML_kernels.H"
#include "BoundaryConditions/PML_current.H"
#include "WarpX_FDTD.H"
#include "WarpXPushFieldsEM_K.H"

#ifdef BL_USE_SENSEI_INSITU
#   include <AMReX_AmrMeshInSituBridge.H>
#endif

#include <AMReX.H>
#include <AMReX_Math.H>
#include <limits>


using namespace amrex;

#ifdef WARPX_USE_PSATD
namespace {
    void
    PushPSATDSinglePatch (
        const int lev,
#ifdef WARPX_DIM_RZ
        SpectralSolverRZ& solver,
#else
        SpectralSolver& solver,
#endif
        std::array<std::unique_ptr<amrex::MultiFab>,3>& Efield,
        std::array<std::unique_ptr<amrex::MultiFab>,3>& Bfield,
        std::array<std::unique_ptr<amrex::MultiFab>,3>& Efield_avg,
        std::array<std::unique_ptr<amrex::MultiFab>,3>& Bfield_avg,
        std::array<std::unique_ptr<amrex::MultiFab>,3>& current,
        std::unique_ptr<amrex::MultiFab>& rho ) {

#ifdef WARPX_DIM_RZ
        amrex::ignore_unused(Efield_avg, Bfield_avg);
#endif

        using Idx = SpectralAvgFieldIndex;

        // Perform forward Fourier transform
#ifdef WARPX_DIM_RZ
        solver.ForwardTransform(lev,
                                *Efield[0], Idx::Ex,
                                *Efield[1], Idx::Ey);
#else
        solver.ForwardTransform(lev, *Efield[0], Idx::Ex);
        solver.ForwardTransform(lev, *Efield[1], Idx::Ey);
#endif
        solver.ForwardTransform(lev, *Efield[2], Idx::Ez);
#ifdef WARPX_DIM_RZ
        solver.ForwardTransform(lev,
                                *Bfield[0], Idx::Bx,
                                *Bfield[1], Idx::By);
#else
        solver.ForwardTransform(lev, *Bfield[0], Idx::Bx);
        solver.ForwardTransform(lev, *Bfield[1], Idx::By);
#endif
        solver.ForwardTransform(lev, *Bfield[2], Idx::Bz);
#ifdef WARPX_DIM_RZ
        solver.ForwardTransform(lev,
                                *current[0], Idx::Jx,
                                *current[1], Idx::Jy);
#else
        solver.ForwardTransform(lev, *current[0], Idx::Jx);
        solver.ForwardTransform(lev, *current[1], Idx::Jy);
#endif
        solver.ForwardTransform(lev, *current[2], Idx::Jz);

        if (rho) {
            solver.ForwardTransform(lev, *rho, Idx::rho_old, 0);
            solver.ForwardTransform(lev, *rho, Idx::rho_new, 1);
        }
#ifdef WARPX_DIM_RZ
        if (WarpX::use_kspace_filter) {
            solver.ApplyFilter(Idx::rho_old);
            solver.ApplyFilter(Idx::rho_new);
            solver.ApplyFilter(Idx::Jx, Idx::Jy, Idx::Jz);
        }
#endif
        // Advance fields in spectral space
        solver.pushSpectralFields();
        // Perform backward Fourier Transform
#ifdef WARPX_DIM_RZ
        solver.BackwardTransform(lev,
                                 *Efield[0], Idx::Ex,
                                 *Efield[1], Idx::Ey);
#else
        solver.BackwardTransform(lev, *Efield[0], Idx::Ex);
        solver.BackwardTransform(lev, *Efield[1], Idx::Ey);
#endif
        solver.BackwardTransform(lev, *Efield[2], Idx::Ez);
#ifdef WARPX_DIM_RZ
        solver.BackwardTransform(lev,
                                 *Bfield[0], Idx::Bx,
                                 *Bfield[1], Idx::By);
#else
        solver.BackwardTransform(lev, *Bfield[0], Idx::Bx);
        solver.BackwardTransform(lev, *Bfield[1], Idx::By);
#endif
        solver.BackwardTransform(lev, *Bfield[2], Idx::Bz);

#ifndef WARPX_DIM_RZ
        if (WarpX::fft_do_time_averaging){
            solver.BackwardTransform(lev, *Efield_avg[0], Idx::Ex_avg);
            solver.BackwardTransform(lev, *Efield_avg[1], Idx::Ey_avg);
            solver.BackwardTransform(lev, *Efield_avg[2], Idx::Ez_avg);

            solver.BackwardTransform(lev, *Bfield_avg[0], Idx::Bx_avg);
            solver.BackwardTransform(lev, *Bfield_avg[1], Idx::By_avg);
            solver.BackwardTransform(lev, *Bfield_avg[2], Idx::Bz_avg);
        }
#endif
    }
}
#endif

void
WarpX::PushPSATD (amrex::Real a_dt)
{
#ifndef WARPX_USE_PSATD
    amrex::ignore_unused(a_dt);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                                     "PushFieldsEM: PSATD solver selected but not built.");
#else
    for (int lev = 0; lev <= finest_level; ++lev) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dt[lev] == a_dt, "dt must be consistent");
        PushPSATD(lev, a_dt);

        // Evolve the fields in the PML boxes
        if (do_pml && pml[lev]->ok()) {
            pml[lev]->PushPSATD(lev);
        }
        ApplyEfieldBoundary(lev,PatchType::fine);
        if (lev > 0) ApplyEfieldBoundary(lev,PatchType::coarse);
        ApplyBfieldBoundary(lev,PatchType::fine);
        if (lev > 0) ApplyBfieldBoundary(lev,PatchType::coarse);
    }
#endif
}

void
WarpX::PushPSATD (int lev, amrex::Real /* dt */) {
#ifndef WARPX_USE_PSATD
    amrex::ignore_unused(lev);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                                     "PushFieldsEM: PSATD solver selected but not built.");
#else
    if (WarpX::maxwell_solver_id != MaxwellSolverAlgo::PSATD) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false,
                                         "WarpX::PushPSATD: only supported for PSATD solver.");
    }
    // Update the fields on the fine and coarse patch
    PushPSATDSinglePatch( lev, *spectral_solver_fp[lev],
        Efield_fp[lev], Bfield_fp[lev], Efield_avg_fp[lev], Bfield_avg_fp[lev], current_fp[lev], rho_fp[lev] );
    if (spectral_solver_cp[lev]) {
        PushPSATDSinglePatch( lev, *spectral_solver_cp[lev],
             Efield_cp[lev], Bfield_cp[lev], Efield_avg_cp[lev], Bfield_avg_cp[lev], current_cp[lev], rho_cp[lev] );
    }

    // Damp the fields in the guard cells along z
    constexpr int zdir = AMREX_SPACEDIM - 1;
    if (WarpX::field_boundary_lo[zdir] == FieldBoundaryType::Damped &&
        WarpX::field_boundary_hi[zdir] == FieldBoundaryType::Damped)
    {
        DampFieldsInGuards(Efield_fp[lev], Bfield_fp[lev]);

        if (WarpX::fft_do_time_averaging)
        {
            DampFieldsInGuards(Efield_avg_fp[lev], Bfield_avg_fp[lev]);
        }
    }
#endif
}

void
WarpX::EvolveB (amrex::Real a_dt)
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        EvolveB(lev, a_dt);
    }
}

void
WarpX::EvolveB (int lev, amrex::Real a_dt)
{
    WARPX_PROFILE("WarpX::EvolveB()");
    EvolveB(lev, PatchType::fine, a_dt);
    if (lev > 0)
    {
        EvolveB(lev, PatchType::coarse, a_dt);
    }
}

void
WarpX::EvolveB (int lev, PatchType patch_type, amrex::Real a_dt)
{

    // Evolve B field in regular cells
    if (patch_type == PatchType::fine) {
        m_fdtd_solver_fp[lev]->EvolveB(Bfield_fp[lev], Efield_fp[lev], G_fp[lev],
                                        m_face_areas[lev], lev, a_dt);
    } else {
        m_fdtd_solver_cp[lev]->EvolveB(Bfield_cp[lev], Efield_cp[lev], G_cp[lev],
                                        m_face_areas[lev], lev, a_dt);
    }

    // Evolve B field in PML cells
    if (do_pml && pml[lev]->ok()) {
        if (patch_type == PatchType::fine) {
            m_fdtd_solver_fp[lev]->EvolveBPML(
                pml[lev]->GetB_fp(), pml[lev]->GetE_fp(), a_dt, WarpX::do_dive_cleaning);
        } else {
            m_fdtd_solver_cp[lev]->EvolveBPML(
                pml[lev]->GetB_cp(), pml[lev]->GetE_cp(), a_dt, WarpX::do_dive_cleaning);
        }
    }

    ApplyBfieldBoundary(lev, patch_type);
}

void
WarpX::ApplySilverMuellerBoundary (amrex::Real a_dt) {
    // Only apply to level 0
    m_fdtd_solver_fp[0]->ApplySilverMuellerBoundary(
        Efield_fp[0], Bfield_fp[0], Geom(0).Domain(), a_dt );
}

void
WarpX::EvolveE (amrex::Real a_dt)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        EvolveE(lev, a_dt);
    }
}

void
WarpX::EvolveE (int lev, amrex::Real a_dt)
{
    WARPX_PROFILE("WarpX::EvolveE()");
    EvolveE(lev, PatchType::fine, a_dt);
    if (lev > 0)
    {
        EvolveE(lev, PatchType::coarse, a_dt);
    }
}

void
WarpX::EvolveE (int lev, PatchType patch_type, amrex::Real a_dt)
{
    // Evolve E field in regular cells
    if (patch_type == PatchType::fine) {
        m_fdtd_solver_fp[lev]->EvolveE(Efield_fp[lev], Bfield_fp[lev],
                                       current_fp[lev], m_edge_lengths[lev],
                                       F_fp[lev], lev, a_dt );
    } else {
        m_fdtd_solver_cp[lev]->EvolveE(Efield_cp[lev], Bfield_cp[lev],
                                       current_cp[lev], m_edge_lengths[lev],
                                       F_cp[lev], lev, a_dt );
    }

    // Evolve E field in PML cells
    if (do_pml && pml[lev]->ok()) {
        if (patch_type == PatchType::fine) {
            m_fdtd_solver_fp[lev]->EvolveEPML(
                pml[lev]->GetE_fp(), pml[lev]->GetB_fp(),
                pml[lev]->Getj_fp(), pml[lev]->GetF_fp(),
                pml[lev]->GetMultiSigmaBox_fp(),
                a_dt, pml_has_particles );
        } else {
            m_fdtd_solver_cp[lev]->EvolveEPML(
                pml[lev]->GetE_cp(), pml[lev]->GetB_cp(),
                pml[lev]->Getj_cp(), pml[lev]->GetF_cp(),
                pml[lev]->GetMultiSigmaBox_cp(),
                a_dt, pml_has_particles );
        }
    }

    ApplyEfieldBoundary(lev, patch_type);

}


void
WarpX::EvolveF (amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_dive_cleaning) return;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        EvolveF(lev, a_dt, a_dt_type);
    }
}

void
WarpX::EvolveF (int lev, amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_dive_cleaning) return;

    EvolveF(lev, PatchType::fine, a_dt, a_dt_type);
    if (lev > 0) EvolveF(lev, PatchType::coarse, a_dt, a_dt_type);
}

void
WarpX::EvolveF (int lev, PatchType patch_type, amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_dive_cleaning) return;

    WARPX_PROFILE("WarpX::EvolveF()");

    const int rhocomp = (a_dt_type == DtType::FirstHalf) ? 0 : 1;

    // Evolve F field in regular cells
    if (patch_type == PatchType::fine) {
        m_fdtd_solver_fp[lev]->EvolveF( F_fp[lev], Efield_fp[lev],
                                        rho_fp[lev], rhocomp, a_dt );
    } else {
        m_fdtd_solver_cp[lev]->EvolveF( F_cp[lev], Efield_cp[lev],
                                        rho_cp[lev], rhocomp, a_dt );
    }

    // Evolve F field in PML cells
    if (do_pml && pml[lev]->ok()) {
        if (patch_type == PatchType::fine) {
            m_fdtd_solver_fp[lev]->EvolveFPML(
                pml[lev]->GetF_fp(), pml[lev]->GetE_fp(), a_dt );
        } else {
            m_fdtd_solver_cp[lev]->EvolveFPML(
                pml[lev]->GetF_cp(), pml[lev]->GetE_cp(), a_dt );
        }
    }
}

void
WarpX::EvolveG (amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_divb_cleaning) return;

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        EvolveG(lev, a_dt, a_dt_type);
    }
}

void
WarpX::EvolveG (int lev, amrex::Real a_dt, DtType a_dt_type)
{
    if (!do_divb_cleaning) return;

    EvolveG(lev, PatchType::fine, a_dt, a_dt_type);

    if (lev > 0)
    {
        EvolveG(lev, PatchType::coarse, a_dt, a_dt_type);
    }
}

void
WarpX::EvolveG (int lev, PatchType patch_type, amrex::Real a_dt, DtType /*a_dt_type*/)
{
    if (!do_divb_cleaning) return;

    WARPX_PROFILE("WarpX::EvolveG()");

    // Evolve G field in regular cells
    if (patch_type == PatchType::fine)
    {
        m_fdtd_solver_fp[lev]->EvolveG(G_fp[lev], Bfield_fp[lev], a_dt);
    }
    else // coarse patch
    {
        m_fdtd_solver_cp[lev]->EvolveG(G_cp[lev], Bfield_cp[lev], a_dt);
    }

    // TODO Evolution in PML cells will go here
}

void
WarpX::MacroscopicEvolveE (amrex::Real a_dt)
{
    for (int lev = 0; lev <= finest_level; ++lev ) {
        MacroscopicEvolveE(lev, a_dt);
    }
}

void
WarpX::MacroscopicEvolveE (int lev, amrex::Real a_dt) {

    WARPX_PROFILE("WarpX::MacroscopicEvolveE()");
    MacroscopicEvolveE(lev, PatchType::fine, a_dt);
    if (lev > 0) {
        amrex::Abort("Macroscopic EvolveE is not implemented for lev>0, yet.");
    }
}

void
WarpX::MacroscopicEvolveE (int lev, PatchType patch_type, amrex::Real a_dt) {
    if (patch_type == PatchType::fine) {
        m_fdtd_solver_fp[lev]->MacroscopicEvolveE( Efield_fp[lev], Bfield_fp[lev],
                                             current_fp[lev], a_dt,
                                             m_macroscopic_properties);
    }
    else {
        amrex::Abort("Macroscopic EvolveE is not implemented for lev > 0, yet.");
    }
    if (do_pml && pml[lev]->ok()) {
        if (patch_type == PatchType::fine) {
            m_fdtd_solver_fp[lev]->EvolveEPML(
                pml[lev]->GetE_fp(), pml[lev]->GetB_fp(),
                pml[lev]->Getj_fp(), pml[lev]->GetF_fp(),
                pml[lev]->GetMultiSigmaBox_fp(),
                a_dt, pml_has_particles );
        } else {
            m_fdtd_solver_cp[lev]->EvolveEPML(
                pml[lev]->GetE_cp(), pml[lev]->GetB_cp(),
                pml[lev]->Getj_cp(), pml[lev]->GetF_cp(),
                pml[lev]->GetMultiSigmaBox_cp(),
                a_dt, pml_has_particles );
        }
    }

    ApplyEfieldBoundary(lev, patch_type);
}

void
WarpX::DampFieldsInGuards(std::array<std::unique_ptr<amrex::MultiFab>,3>& Efield,
                          std::array<std::unique_ptr<amrex::MultiFab>,3>& Bfield) {

    constexpr int zdir = (AMREX_SPACEDIM - 1);

    for ( amrex::MFIter mfi(*Efield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        amrex::Array4<amrex::Real> const& Ex_arr = Efield[0]->array(mfi);
        amrex::Array4<amrex::Real> const& Ey_arr = Efield[1]->array(mfi);
        amrex::Array4<amrex::Real> const& Ez_arr = Efield[2]->array(mfi);
        amrex::Array4<amrex::Real> const& Bx_arr = Bfield[0]->array(mfi);
        amrex::Array4<amrex::Real> const& By_arr = Bfield[1]->array(mfi);
        amrex::Array4<amrex::Real> const& Bz_arr = Bfield[2]->array(mfi);

        // Get the tileboxes from Efield and Bfield so that they include the guard cells
        // and take the staggering of each MultiFab into account
        amrex::Box tex = amrex::convert((*Efield[0])[mfi].box(), Efield[0]->ixType().toIntVect());
        amrex::Box tey = amrex::convert((*Efield[1])[mfi].box(), Efield[1]->ixType().toIntVect());
        amrex::Box tez = amrex::convert((*Efield[2])[mfi].box(), Efield[2]->ixType().toIntVect());
        amrex::Box tbx = amrex::convert((*Bfield[0])[mfi].box(), Bfield[0]->ixType().toIntVect());
        amrex::Box tby = amrex::convert((*Bfield[1])[mfi].box(), Bfield[1]->ixType().toIntVect());
        amrex::Box tbz = amrex::convert((*Bfield[2])[mfi].box(), Bfield[2]->ixType().toIntVect());

        // Get smallEnd of tileboxes
        const int tex_smallEnd_z = tex.smallEnd(zdir);
        const int tey_smallEnd_z = tey.smallEnd(zdir);
        const int tez_smallEnd_z = tez.smallEnd(zdir);
        const int tbx_smallEnd_z = tbx.smallEnd(zdir);
        const int tby_smallEnd_z = tby.smallEnd(zdir);
        const int tbz_smallEnd_z = tbz.smallEnd(zdir);

        // Get bigEnd of tileboxes
        const int tex_bigEnd_z = tex.bigEnd(zdir);
        const int tey_bigEnd_z = tey.bigEnd(zdir);
        const int tez_bigEnd_z = tez.bigEnd(zdir);
        const int tbx_bigEnd_z = tbx.bigEnd(zdir);
        const int tby_bigEnd_z = tby.bigEnd(zdir);
        const int tbz_bigEnd_z = tbz.bigEnd(zdir);

        // Box for the whole simulation domain
        amrex::Box const& domain = Geom(0).Domain();
        int const nz_domain = domain.bigEnd(zdir);

        // Set the tileboxes so that they only cover the lower/upper half of the guard cells
        constrain_tilebox_to_guards(tex, zdir, nz_domain, tex_smallEnd_z, tex_bigEnd_z);
        constrain_tilebox_to_guards(tey, zdir, nz_domain, tey_smallEnd_z, tey_bigEnd_z);
        constrain_tilebox_to_guards(tez, zdir, nz_domain, tez_smallEnd_z, tez_bigEnd_z);
        constrain_tilebox_to_guards(tbx, zdir, nz_domain, tbx_smallEnd_z, tbx_bigEnd_z);
        constrain_tilebox_to_guards(tby, zdir, nz_domain, tby_smallEnd_z, tby_bigEnd_z);
        constrain_tilebox_to_guards(tbz, zdir, nz_domain, tbz_smallEnd_z, tbz_bigEnd_z);

        amrex::ParallelFor(
            tex, Efield[0]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
            {
                damp_field_in_guards(Ex_arr, i, j, k, icomp, zdir, nz_domain, tex_smallEnd_z, tex_bigEnd_z);
            },
            tey, Efield[1]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
            {
                damp_field_in_guards(Ey_arr, i, j, k, icomp, zdir, nz_domain, tey_smallEnd_z, tey_bigEnd_z);
            },
            tez, Efield[2]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
            {
                damp_field_in_guards(Ez_arr, i, j, k, icomp, zdir, nz_domain, tez_smallEnd_z, tez_bigEnd_z);
            }
        );

        amrex::ParallelFor(
            tbx, Bfield[0]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
            {
                damp_field_in_guards(Bx_arr, i, j, k, icomp, zdir, nz_domain, tbx_smallEnd_z, tbx_bigEnd_z);
            },
            tby, Bfield[1]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
            {
                damp_field_in_guards(By_arr, i, j, k, icomp, zdir, nz_domain, tby_smallEnd_z, tby_bigEnd_z);
            },
            tbz, Bfield[2]->nComp(), [=] AMREX_GPU_DEVICE (int i, int j, int k, int icomp)
            {
                damp_field_in_guards(Bz_arr, i, j, k, icomp, zdir, nz_domain, tbz_smallEnd_z, tbz_bigEnd_z);
            }
        );
    }
}

#ifdef WARPX_DIM_RZ
// This scales the current by the inverse volume and wraps around the depostion at negative radius.
// It is faster to apply this on the grid than to do it particle by particle.
// It is put here since there isn't another nice place for it.
void
WarpX::ApplyInverseVolumeScalingToCurrentDensity (MultiFab* Jx, MultiFab* Jy, MultiFab* Jz, int lev)
{
    const long ngJ = Jx->nGrow();
    const std::array<Real,3>& dx = WarpX::CellSize(lev);
    const Real dr = dx[0];

    constexpr int NODE = amrex::IndexType::NODE;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Jx->ixType().toIntVect()[0] != NODE,
        "Jr should never node-centered in r");


    for ( MFIter mfi(*Jx, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {

        Array4<Real> const& Jr_arr = Jx->array(mfi);
        Array4<Real> const& Jt_arr = Jy->array(mfi);
        Array4<Real> const& Jz_arr = Jz->array(mfi);

        Box const & tilebox = mfi.tilebox();
        Box tbr = convert( tilebox, Jx->ixType().toIntVect() );
        Box tbt = convert( tilebox, Jy->ixType().toIntVect() );
        Box tbz = convert( tilebox, Jz->ixType().toIntVect() );

        // Lower corner of tile box physical domain
        // Note that this is done before the tilebox.grow so that
        // these do not include the guard cells.
        std::array<amrex::Real,3> galilean_shift = {0,0,0};
        const std::array<Real, 3>& xyzmin = WarpX::LowerCorner(tilebox, galilean_shift, lev);
        const Real rmin  = xyzmin[0];
        const Real rminr = xyzmin[0] + (tbr.type(0) == NODE ? 0. : 0.5*dx[0]);
        const Real rmint = xyzmin[0] + (tbt.type(0) == NODE ? 0. : 0.5*dx[0]);
        const Real rminz = xyzmin[0] + (tbz.type(0) == NODE ? 0. : 0.5*dx[0]);
        const Dim3 lo = lbound(tilebox);
        const int irmin = lo.x;

        // For ishift, 1 means cell centered, 0 means node centered
        int const ishift_t = (rmint > rmin ? 1 : 0);
        int const ishift_z = (rminz > rmin ? 1 : 0);

        const int nmodes = n_rz_azimuthal_modes;

        // Grow the tileboxes to include the guard cells, except for the
        // guard cells at negative radius.
        if (rmin > 0.) {
           tbr.growLo(0, ngJ);
           tbt.growLo(0, ngJ);
           tbz.growLo(0, ngJ);
        }
        tbr.growHi(0, ngJ);
        tbt.growHi(0, ngJ);
        tbz.growHi(0, ngJ);
        tbr.grow(1, ngJ);
        tbt.grow(1, ngJ);
        tbz.grow(1, ngJ);

        // Rescale current in r-z mode since the inverse volume factor was not
        // included in the current deposition.
        amrex::ParallelFor(tbr, tbt, tbz,
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            // Wrap the current density deposited in the guard cells around
            // to the cells above the axis.
            // Note that Jr(i==0) is at 1/2 dr.
            if (rmin == 0. && 0 <= i && i < ngJ) {
                Jr_arr(i,j,0,0) -= Jr_arr(-1-i,j,0,0);
            }
            // Apply the inverse volume scaling
            // Since Jr is never node centered in r, no need for distinction
            // between on axis and off-axis factors
            const amrex::Real r = amrex::Math::abs(rminr + (i - irmin)*dr);
            Jr_arr(i,j,0,0) /= (2.*MathConst::pi*r);

            for (int imode=1 ; imode < nmodes ; imode++) {
                // Wrap the current density deposited in the guard cells around
                // to the cells above the axis.
                // Note that Jr(i==0) is at 1/2 dr.
                if (rmin == 0. && 0 <= i && i < ngJ) {
                    Jr_arr(i,j,0,2*imode-1) -= Jr_arr(-1-i,j,0,2*imode-1);
                    Jr_arr(i,j,0,2*imode) -= Jr_arr(-1-i,j,0,2*imode);
                }
                // Apply the inverse volume scaling
                // Since Jr is never node centered in r, no need for distinction
                // between on axis and off-axis factors
                Jr_arr(i,j,0,2*imode-1) /= (2.*MathConst::pi*r);
                Jr_arr(i,j,0,2*imode) /= (2.*MathConst::pi*r);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            // Wrap the current density deposited in the guard cells around
            // to the cells above the axis.
            // If Jt is node centered, Jt[0] is located on the boundary.
            // If Jt is cell centered, Jt[0] is at 1/2 dr.
            if (rmin == 0. && 1-ishift_t <= i && i <= ngJ-ishift_t) {
                Jt_arr(i,j,0,0) -= Jt_arr(-ishift_t-i,j,0,0);
            }

            // Apply the inverse volume scaling
            // Jt is forced to zero on axis.
            const amrex::Real r = amrex::Math::abs(rmint + (i - irmin)*dr);
            if (r == 0.) {
                Jt_arr(i,j,0,0) = 0.;
            } else {
                Jt_arr(i,j,0,0) /= (2.*MathConst::pi*r);
            }

            for (int imode=1 ; imode < nmodes ; imode++) {
                // Wrap the current density deposited in the guard cells around
                // to the cells above the axis.
                if (rmin == 0. && 1-ishift_t <= i && i <= ngJ-ishift_t) {
                    Jt_arr(i,j,0,2*imode-1) -= Jt_arr(-ishift_t-i,j,0,2*imode-1);
                    Jt_arr(i,j,0,2*imode) -= Jt_arr(-ishift_t-i,j,0,2*imode);
                }

                // Apply the inverse volume scaling
                // Jt is forced to zero on axis.
                if (r == 0.) {
                    Jt_arr(i,j,0,2*imode-1) = 0.;
                    Jt_arr(i,j,0,2*imode) = 0.;
                } else {
                    Jt_arr(i,j,0,2*imode-1) /= (2.*MathConst::pi*r);
                    Jt_arr(i,j,0,2*imode) /= (2.*MathConst::pi*r);
                }
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/)
        {
            // Wrap the current density deposited in the guard cells around
            // to the cells above the axis.
            // If Jz is node centered, Jt[0] is located on the boundary.
            // If Jz is cell centered, Jt[0] is at 1/2 dr.
            if (rmin == 0. && 1-ishift_z <= i && i <= ngJ-ishift_z) {
                Jz_arr(i,j,0,0) -= Jz_arr(-ishift_z-i,j,0,0);
            }

            // Apply the inverse volume scaling
            const amrex::Real r = amrex::Math::abs(rminz + (i - irmin)*dr);
            if (r == 0.) {
                // Verboncoeur JCP 164, 421-427 (2001) : corrected volume on axis
                Jz_arr(i,j,0,0) /= (MathConst::pi*dr/3.);
            } else {
                Jz_arr(i,j,0,0) /= (2.*MathConst::pi*r);
            }

            for (int imode=1 ; imode < nmodes ; imode++) {
                // Wrap the current density deposited in the guard cells around
                // to the cells above the axis.
                if (rmin == 0. && 1-ishift_z <= i && i <= ngJ-ishift_z) {
                    Jz_arr(i,j,0,2*imode-1) -= Jz_arr(-ishift_z-i,j,0,2*imode-1);
                    Jz_arr(i,j,0,2*imode) -= Jz_arr(-ishift_z-i,j,0,2*imode);
                }

                // Apply the inverse volume scaling
                if (r == 0.) {
                    // Verboncoeur JCP 164, 421-427 (2001) : corrected volume on axis
                    Jz_arr(i,j,0,2*imode-1) /= (MathConst::pi*dr/3.);
                    Jz_arr(i,j,0,2*imode) /= (MathConst::pi*dr/3.);
                } else {
                    Jz_arr(i,j,0,2*imode-1) /= (2.*MathConst::pi*r);
                    Jz_arr(i,j,0,2*imode) /= (2.*MathConst::pi*r);
                }
            }

        });
    }
}

void
WarpX::ApplyInverseVolumeScalingToChargeDensity (MultiFab* Rho, int lev)
{
    const long ngRho = Rho->nGrow();
    const std::array<Real,3>& dx = WarpX::CellSize(lev);
    const Real dr = dx[0];

    constexpr int NODE = amrex::IndexType::NODE;

    Box tilebox;

    for ( MFIter mfi(*Rho, TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {

        Array4<Real> const& Rho_arr = Rho->array(mfi);

        tilebox = mfi.tilebox();
        Box tb = convert( tilebox, Rho->ixType().toIntVect() );

        // Lower corner of tile box physical domain
        // Note that this is done before the tilebox.grow so that
        // these do not include the guard cells.
        std::array<amrex::Real,3> galilean_shift = {0,0,0};
        const std::array<Real, 3>& xyzmin = WarpX::LowerCorner(tilebox, galilean_shift, lev);
        const Dim3 lo = lbound(tilebox);
        const Real rmin = xyzmin[0];
        const Real rminr = xyzmin[0] + (tb.type(0) == NODE ? 0. : 0.5*dx[0]);
        const int irmin = lo.x;
        int ishift = (rminr > rmin ? 1 : 0);

        // Grow the tilebox to include the guard cells, except for the
        // guard cells at negative radius.
        if (rmin > 0.) {
           tb.growLo(0, ngRho);
        }
        tb.growHi(0, ngRho);
        tb.grow(1, ngRho);

        // Rescale charge in r-z mode since the inverse volume factor was not
        // included in the charge deposition.
        // Note that the loop is also over ncomps, which takes care of the RZ modes,
        // as well as the old and new rho.
        amrex::ParallelFor(tb, Rho->nComp(),
        [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/, int icomp)
        {
            // Wrap the charge density deposited in the guard cells around
            // to the cells above the axis.
            // Rho is located on the boundary
            if (rmin == 0. && 1-ishift <= i && i <= ngRho-ishift) {
                Rho_arr(i,j,0,icomp) -= Rho_arr(-ishift-i,j,0,icomp);
            }

            // Apply the inverse volume scaling
            const amrex::Real r = amrex::Math::abs(rminr + (i - irmin)*dr);
            if (r == 0.) {
                // Verboncoeur JCP 164, 421-427 (2001) : corrected volume on axis
                Rho_arr(i,j,0,icomp) /= (MathConst::pi*dr/3.);
            } else {
                Rho_arr(i,j,0,icomp) /= (2.*MathConst::pi*r);
            }
        });
    }
}
#endif

void
WarpX::EvolveRIP (amrex::Real dt, bool half)
{
    // if half=true , E and B are pushed by one time step, from n-1/2 to n+1/2
    // if half=false, E and B are pushed by one time step, from n     to n+1

    Real constexpr c = PhysConst::c;
    Real constexpr mu0 = PhysConst::mu0;

    // Loop over mesh-refinement levels
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        // Get the field data structures
        std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield =
            half ? Efield_fp_half[lev] : Efield_fp[lev];
        std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Bfield =
            half ? Bfield_fp_half[lev] : Bfield_fp[lev];
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfield = current_fp[lev];

        std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Efield_mid =
            half ? Efield_fp[lev] : Efield_fp_half[lev];
        std::array< std::unique_ptr<amrex::MultiFab>, 3 >& Bfield_mid =
            half ? Bfield_fp[lev] : Bfield_fp_half[lev];
        std::array< std::unique_ptr<amrex::MultiFab>, 3 > const& Jfieldo = current_fp_old[lev];

        // Get parameters of cells
        std::array<Real,3> dx = CellSize(lev);

        amrex::Print()<<"dx[0] = " << dx[0]<<'\n';
        amrex::Print()<<"dx[1] = " << dx[1]<<'\n';
        amrex::Print()<<"dx[2] = " << dx[2]<<'\n';

        // Temporary fields to update E and B by 1 time step, using the middle value
        std::cout<<Efield[0]->boxArray()<<'\n';
        amrex::MultiFab mf_Ex_tmp(Efield[0]->boxArray(), Efield[0]->DistributionMap(), 1, Efield[0]->nGrowVect());
        amrex::MultiFab mf_Ey_tmp(Efield[1]->boxArray(), Efield[1]->DistributionMap(), 1, Efield[1]->nGrowVect());
        amrex::MultiFab mf_Ez_tmp(Efield[2]->boxArray(), Efield[2]->DistributionMap(), 1, Efield[2]->nGrowVect());
        amrex::MultiFab mf_Bx_tmp(Bfield[0]->boxArray(), Bfield[0]->DistributionMap(), 1, Bfield[0]->nGrowVect());
        amrex::MultiFab mf_By_tmp(Bfield[1]->boxArray(), Bfield[1]->DistributionMap(), 1, Bfield[1]->nGrowVect());
        amrex::MultiFab mf_Bz_tmp(Bfield[2]->boxArray(), Bfield[2]->DistributionMap(), 1, Bfield[2]->nGrowVect());
        amrex::MultiFab::Copy(mf_Ex_tmp, *Efield[0], 0, 0, 1, Efield[0]->nGrowVect());
        amrex::MultiFab::Copy(mf_Ey_tmp, *Efield[1], 0, 0, 1, Efield[1]->nGrowVect());
        amrex::MultiFab::Copy(mf_Ez_tmp, *Efield[2], 0, 0, 1, Efield[2]->nGrowVect());
        amrex::MultiFab::Copy(mf_Bx_tmp, *Bfield[0], 0, 0, 1, Bfield[0]->nGrowVect());
        amrex::MultiFab::Copy(mf_By_tmp, *Bfield[1], 0, 0, 1, Bfield[1]->nGrowVect());
        amrex::MultiFab::Copy(mf_Bz_tmp, *Bfield[2], 0, 0, 1, Bfield[2]->nGrowVect());
        //mf_Ex_tmp.setVal(0.);
        //mf_Ey_tmp.setVal(0.);
        //mf_Ez_tmp.setVal(0.);
        //mf_Bx_tmp.setVal(0.);
        //mf_By_tmp.setVal(0.);
        //mf_Bz_tmp.setVal(0.);

// Updating E and B at time step n+1/2
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        // Loop over boxes on this rank
        std::cout << "Before Loop MFI" << std::endl ;
        for ( MFIter mfi(*Efield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi )  {

            std::cout << "Loop MFI" << std::endl ;

            // Extract field data for this grid/tile

            // E and B that we are evolving by 1 time step
            // if half: E and B are given at n-1/2
            // if not : E and B are given at n
            Array4<Real> const& Ex = Efield[0]->array(mfi);
            Array4<Real> const& Ey = Efield[1]->array(mfi);
            Array4<Real> const& Ez = Efield[2]->array(mfi);

            Array4<Real> const& Bx = Bfield[0]->array(mfi);
            Array4<Real> const& By = Bfield[1]->array(mfi);
            Array4<Real> const& Bz = Bfield[2]->array(mfi);

            Array4<Real> const& jx = Jfield[0]->array(mfi);
            Array4<Real> const& jy = Jfield[1]->array(mfi);
            Array4<Real> const& jz = Jfield[2]->array(mfi);

            Array4<Real> const& jxo = Jfieldo[0]->array(mfi);
            Array4<Real> const& jyo = Jfieldo[1]->array(mfi);
            Array4<Real> const& jzo = Jfieldo[2]->array(mfi);

            // E and B in the middle of the current step
            // if half: Em and Bm are given at n
            // if not : Em and Bm are given at n+1/2
            Array4<Real> const& Exm = Efield_mid[0]->array(mfi);
            Array4<Real> const& Eym = Efield_mid[1]->array(mfi);
            Array4<Real> const& Ezm = Efield_mid[2]->array(mfi);

            Array4<Real> const& Bxm = Bfield_mid[0]->array(mfi);
            Array4<Real> const& Bym = Bfield_mid[1]->array(mfi);
            Array4<Real> const& Bzm = Bfield_mid[2]->array(mfi);

            // Extract tileboxes for which to loop
            Box const& tex  = mfi.tilebox(Efield[0]->ixType().toIntVect());
            Box const& tey  = mfi.tilebox(Efield[1]->ixType().toIntVect());
            Box const& tez  = mfi.tilebox(Efield[2]->ixType().toIntVect());

            Box const& tbx  = mfi.tilebox(Bfield[0]->ixType().toIntVect());
            Box const& tby  = mfi.tilebox(Bfield[1]->ixType().toIntVect());
            Box const& tbz  = mfi.tilebox(Bfield[2]->ixType().toIntVect());

            Array4<Real> const& Ex_tmp = mf_Ex_tmp.array(mfi);
            Array4<Real> const& Ey_tmp = mf_Ey_tmp.array(mfi);
            Array4<Real> const& Ez_tmp = mf_Ez_tmp.array(mfi);
            Array4<Real> const& Bx_tmp = mf_Bx_tmp.array(mfi);
            Array4<Real> const& By_tmp = mf_By_tmp.array(mfi);
            Array4<Real> const& Bz_tmp = mf_Bz_tmp.array(mfi);

            // Loops over the cells and update the fields
            // Push E by 1 time step, using the middle value as a source term
            amrex::ParallelFor(
                tex, tey, tez,

                [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if defined WARPX_DIM_3D

                    amrex::Real gamma_x_p = -c*mu0*(jx(i+1,j  ,k+1)+jxo(i+1,j  ,k+1)+jx(i+1,j  ,k  )+jxo(i+1,j  ,k  ))/4 + c*(Bz(i+1,j+1,k  )+Bz(i+1,j+1,k+1)-Bz(i+1,j  ,k  )-Bz(i+1,j  ,k+1))/(2*dx[1]) ;
                    amrex::Real gamma_x_m = -c*mu0*(jx(i+1,j  ,k  )+jxo(i+1,j  ,k  )+jx(i+1,j  ,k-1)+jxo(i+1,j  ,k-1))/4 + c*(Bz(i+1,j+1,k-1)+Bz(i+1,j+1,k  )-Bz(i+1,j  ,k-1)-Bz(i+1,j  ,k  ))/(2*dx[1]) ;

                    amrex::Real phi_y_p = (Ez(i+1,j  ,k  )+Ez(i+1,j  ,k+1)-Ez(i  ,j  ,k  )-Ez(i  ,j  ,k+1))/(2*dx[0]) ;
                    amrex::Real phi_y_m = (Ez(i+1,j  ,k-1)+Ez(i+1,j  ,k  )-Ez(i  ,j  ,k-1)-Ez(i  ,j  ,k  ))/(2*dx[0]) ;

                    Exh_tmp(i,j,k) = (Exh(i,j,k-1) + Exh(i,j,k+1))/2 - c*(Byh(i,j,k+1) - Byh(i,j,k-1))/2 + (gamma_x_m + phi_y_m + gamma_x_p - phi_y_p)*dx[2]/2 ;

#elif defined WARPX_DIM_XZ

                    const amrex::Real gamma_x_p = half
                        ? -c*mu0*(jx(i+1,j+1,0) + jxo(i+1,j+1,0) + jx(i+1,j  ,0) + jxo(i+1,j  ,0))/4._rt
                        : -c*mu0*(jx(i+1,j+1,0) +                  jx(i+1,j  ,0)                 )/2._rt;
                    const amrex::Real gamma_x_m = half
                        ? -c*mu0*(jx(i+1,j  ,0) + jxo(i+1,j  ,0) + jx(i+1,j-1,0) + jxo(i+1,j-1,0))/4._rt
                        : -c*mu0*(jx(i+1,j  ,0) +                  jx(i+1,j-1,0)                 )/2._rt;

                    const amrex::Real phi_y_p = (Ezm(i+1,j  ,0)+Ezm(i+1,j+1,0)-Ezm(i  ,j  ,0)-Ezm(i  ,j+1,0))/(2._rt*dx[0]) ;
                    const amrex::Real phi_y_m = (Ezm(i+1,j-1,0)+Ezm(i+1,j  ,0)-Ezm(i  ,j-1,0)-Ezm(i  ,j  ,0))/(2._rt*dx[0]) ;

                    Ex(i,j,0) = (Ex_tmp(i,j-1,0) + Ex_tmp(i,j+1,0))/2 - c*(By_tmp(i,j+1,0) - By_tmp(i,j-1,0))/2._rt + (gamma_x_m + phi_y_m + gamma_x_p - phi_y_p)*dx[2]/2._rt;
                    amrex::ignore_unused(k);
#endif

                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if defined WARPX_DIM_3D

                    const amrex::Real gamma_y_p = half
                        ? -c*mu0*(jy(i  ,j+1,k  )+jyo(i  ,j+1,k  )+jy(i  ,j+1,k+1)+jyo(i  ,j+1,k+1))/4 - c*(Bzm(i+1,j+1,k  )+Bzm(i+1,j+1,k+1)-Bzm(i  ,j+1,k  )-Bzm(i  ,j+1,k+1))/(2*dx[0])
                        : -c*mu0*(jy(i  ,j+1,k  )+                 jy(i  ,j+1,k+1)                 )/2 - c*(Bzm(i+1,j+1,k  )+Bzm(i+1,j+1,k+1)-Bzm(i  ,j+1,k  )-Bzm(i  ,j+1,k+1))/(2*dx[0]);
                    const amrex::Real gamma_y_m = half
                        ? -c*mu0*(jy(i  ,j+1,k-1)+jyo(i  ,j+1,k-1)+jy(i  ,j+1,k  )+jyo(i  ,j+1,k  ))/4 - c*(Bzm(i+1,j+1,k-1)+Bzm(i+1,j+1,k  )-Bzm(i  ,j+1,k-1)-Bzm(i  ,j+1,k  ))/(2*dx[0])
                        : -c*mu0*(jy(i  ,j+1,k-1)+                +jy(i  ,j+1,k  )                 )/2 - c*(Bzm(i+1,j+1,k-1)+Bzm(i+1,j+1,k  )-Bzm(i  ,j+1,k-1)-Bzm(i  ,j+1,k  ))/(2*dx[0]);

                    const amrex::Real phi_x_p = -(Ezm(i  ,j+1,k  )+Ezm(i  ,j+1,k+1)-Ezm(i  ,j  ,k  )-Ezm(i  ,j  ,k+1))/(2*dx[1]) ;
                    const amrex::Real phi_x_m = -(Ezm(i  ,j+1,k-1)+Ezm(i  ,j+1,k  )-Ezm(i  ,j  ,k-1)-Ezm(i  ,j  ,k  ))/(2*dx[1]) ;

                    Ey(i,j,k) =  (Ey_tmp(i,j,k-1) + Ey_tmp(i,j,k+1))/2 + c*(Bx_tmp(i,j,k+1) - Bx_tmp(i,j,k-1))/2 + (gamma_y_m - phi_x_m + gamma_y_p + phi_x_p)*dx[2]/2  ;

#elif defined WARPX_DIM_XZ
                    const amrex::Real gamma_y_p = half
                        ? -c*mu0*(jy(i  ,j  ,0)+jyo(i  ,j  ,0)+jy(i  ,j+1,0)+jyo(i  ,j+1,0))/4 - c*(Bzm(i+1,j  ,0)+Bzm(i+1,j+1,0)-Bzm(i  ,j  ,0)-Bzm(i  ,j+1,0))/(2*dx[0])
                        : -c*mu0*(jy(i  ,j  ,0)+               jy(i  ,j+1,0)               )/2 - c*(Bzm(i+1,j  ,0)+Bzm(i+1,j+1,0)-Bzm(i  ,j  ,0)-Bzm(i  ,j+1,0))/(2*dx[0]);
                    const amrex::Real gamma_y_m = half
                        ? -c*mu0*(jy(i  ,j-1,0)+jyo(i  ,j-1,0)+jy(i  ,j  ,0)+jyo(i  ,j  ,0))/4 - c*(Bzm(i+1,j-1,0)+Bzm(i+1,j  ,0)-Bzm(i  ,j-1,0)-Bzm(i  ,j  ,0))/(2*dx[0])
                        : -c*mu0*(jy(i  ,j-1,0)+               jy(i  ,j  ,0)               )/2 - c*(Bzm(i+1,j-1,0)+Bzm(i+1,j  ,0)-Bzm(i  ,j-1,0)-Bzm(i  ,j  ,0))/(2*dx[0]);

                    Ey(i,j,0) =  (Ey_tmp(i,j-1,0) + Ey_tmp(i,j+1,0))/2 + c*(Bx_tmp(i,j+1,0) - Bx_tmp(i,j-1,0))/2 + (gamma_y_m + gamma_y_p)*dx[2]/2  ;
                    amrex::ignore_unused(k);

#endif

                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if defined WARPX_DIM_3D

                    amrex::Real gamma_z = - c*mu0*(jz(i,j,k)+jzo(i,j,k))/2 + c*(By(i+1,j,k)-By(i,j,k))/dx[0] - c*(Bx(i,j+1,k)-Bx(i,j,k))/dx[1] ;
                    Ezh_tmp(i,j,k) = Ezh(i,j,k) + dx[2]*gamma_z ;

#elif defined WARPX_DIM_XZ
                    amrex::Real gamma_z = half
                        ? - c*mu0*(jz(i,j,0)+jzo(i,j,0))/2._rt + c*(Bym(i+1,j,0)-Bym(i,j,0))/dx[0]
                        : - c*mu0*(jz(i,j,0)           )       + c*(Bym(i+1,j,0)-Bym(i,j,0))/dx[0];
                    Ez(i,j,0) = Ez_tmp(i,j,0) + dx[2]*gamma_z ;
                    amrex::ignore_unused(k);
#endif
                }
                );

            // Field B at time step n+1/2, depending on fields E and B at time steps n and n-1/2 :
            amrex::ParallelFor(
                tbx, tby, tbz,

                [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if defined WARPX_DIM_3D

                    amrex::Real gamma_y_p = -c*mu0*(jy(i  ,j+1,k  )+jyo(i  ,j+1,k  )+jy(i  ,j+1,k+1)+jyo(i  ,j+1,k+1))/4 - c*(Bz(i+1,j+1,k  )+Bz(i+1,j+1,k+1)-Bz(i  ,j+1,k  )-Bz(i  ,j+1,k+1))/(2*dx[0]) ;
                    amrex::Real gamma_y_m = -c*mu0*(jy(i  ,j+1,k-1)+jyo(i  ,j+1,k-1)+jy(i  ,j+1,k  )+jyo(i  ,j+1,k  ))/4 - c*(Bz(i+1,j+1,k-1)+Bz(i+1,j+1,k  )-Bz(i  ,j+1,k-1)-Bz(i  ,j+1,k  ))/(2*dx[0]) ;

                    amrex::Real phi_x_p = -(Ez(i  ,j+1,k  )+Ez(i  ,j+1,k+1)-Ez(i  ,j  ,k  )-Ez(i  ,j  ,k+1))/(2*dx[1]) ;
                    amrex::Real phi_x_m = -(Ez(i  ,j+1,k-1)+Ez(i  ,j+1,k  )-Ez(i  ,j  ,k-1)-Ez(i  ,j  ,k  ))/(2*dx[1]) ;

                    Bxh_tmp(i, j, k) = (Bxh(i,j,k-1) + Bxh(i,j,k+1))/2 + (Eyh(i,j,k+1) - Eyh(i,j,k-1))/2/c + (-gamma_y_m + phi_x_m + gamma_y_p + phi_x_p)*dx[2]/2/c ;

 #elif defined WARPX_DIM_XZ

                    amrex::Real gamma_y_p = half
                        ? -c*mu0*(jy(i  ,j  ,0)+jyo(i  ,j  ,0)+jy(i  ,j+1,0)+jyo(i  ,j+1,0))/4 - c*(Bzm(i+1,j  ,0)+Bzm(i+1,j+1,0)-Bzm(i  ,j  ,0)-Bzm(i  ,j+1,0))/(2*dx[0])
                        : -c*mu0*(jy(i  ,j  ,0)+               jy(i  ,j+1,0)               )/2 - c*(Bzm(i+1,j  ,0)+Bzm(i+1,j+1,0)-Bzm(i  ,j  ,0)-Bzm(i  ,j+1,0))/(2*dx[0]);
                    amrex::Real gamma_y_m = half
                        ? -c*mu0*(jy(i  ,j-1,0)+jyo(i  ,j-1,0)+jy(i  ,j  ,0)+jyo(i  ,j  ,0))/4 - c*(Bzm(i+1,j-1,0)+Bzm(i+1,j  ,0)-Bzm(i  ,j-1,0)-Bzm(i  ,j  ,0))/(2*dx[0])
                        : -c*mu0*(jy(i  ,j-1,0)+               jy(i  ,j  ,0)               )/2 - c*(Bzm(i+1,j-1,0)+Bzm(i+1,j  ,0)-Bzm(i  ,j-1,0)-Bzm(i  ,j  ,0))/(2*dx[0]);

                    Bx(i,j,0) = (Bx_tmp(i,j-1,0) + Bx_tmp(i,j+1,0))/2 + (Ey_tmp(i,j+1,0) - Ey_tmp(i,j-1,0))/2/c + (-gamma_y_m + gamma_y_p)*dx[2]/2/c ;

                    amrex::ignore_unused(k);

#endif

                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if defined WARPX_DIM_3D

                    amrex::Real gamma_x_p = -c*mu0*(jx(i+1,j  ,k+1)+jxo(i+1,j  ,k+1)+jx(i+1,j  ,k  )+jxo(i+1,j  ,k  ))/4 + c*(Bz(i+1,j+1,k  )+Bz(i+1,j+1,k+1)-Bz(i+1,j  ,k  )-Bz(i+1,j  ,k+1))/(2*dx[1]) ;
                    amrex::Real gamma_x_m = -c*mu0*(jx(i+1,j  ,k  )+jxo(i+1,j  ,k  )+jx(i+1,j  ,k-1)+jxo(i+1,j  ,k-1))/4 + c*(Bz(i+1,j+1,k-1)+Bz(i+1,j+1,k  )-Bz(i+1,j  ,k-1)-Bz(i+1,j  ,k  ))/(2*dx[1]) ;

                    amrex::Real phi_y_p = (Ez(i+1,j  ,k  )+Ez(i+1,j  ,k+1)-Ez(i  ,j  ,k  )-Ez(i  ,j  ,k+1))/(2*dx[0]) ;
                    amrex::Real phi_y_m = (Ez(i+1,j  ,k-1)+Ez(i+1,j  ,k  )-Ez(i  ,j  ,k-1)-Ez(i  ,j  ,k  ))/(2*dx[0]) ;

                    Byh_tmp(i, j, k) = c*(Byh(i,j,k-1) + Byh(i,j,k+1))/2 - (Exh(i,j,k+1) - Exh(i,j,k-1))/2 + (gamma_x_m + phi_y_m - gamma_x_p + phi_y_p)*dx[2]/2 ;

#elif defined WARPX_DIM_XZ

                    amrex::Real gamma_x_p = half
                        ? -c*mu0*(jx(i+1,j+1,0)+jxo(i+1,j+1,0)+jx(i+1,j  ,0)+jxo(i+1,j  ,0))/4
                        : -c*mu0*(jx(i+1,j+1,0)+               jx(i+1,j  ,0)               )/2;
                    amrex::Real gamma_x_m = half
                        ? -c*mu0*(jx(i+1,j  ,0)+jxo(i+1,j  ,0)+jx(i+1,j-1,0)+jxo(i+1,j-1,0))/4
                        : -c*mu0*(jx(i+1,j  ,0)+               jx(i+1,j-1,0)               )/2;

                    amrex::Real phi_y_p = (Ezm(i+1,j  ,0)+Ezm(i+1,j+1,0)-Ezm(i  ,j  ,0)-Ezm(i  ,j+1,0))/(2*dx[0]) ;
                    amrex::Real phi_y_m = (Ezm(i+1,j-1,0)+Ezm(i+1,j  ,0)-Ezm(i  ,j-1,0)-Ezm(i  ,j  ,0))/(2*dx[0]) ;

                    By(i,j,0) = (By_tmp(i,j-1,0) + By_tmp(i,j+1,0))/2 - (Ex_tmp(i,j+1,0) - Ex_tmp(i,j-1,0))/2/c + (gamma_x_m + phi_y_m - gamma_x_p + phi_y_p)*dx[2]/2/c ;

                    amrex::ignore_unused(k);

#endif

                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if defined WARPX_DIM_3D

                    amrex::Real phi_z = (Exm(i+1,j+1,k) - Exm(i+1,j,k))/dx[1] - (Eym(i+1,j+1,k) - Eym(i,j+1,k))/dx[0] ;
                    Bz_tmp(i, j, k) = Bz(i,j,k) + dx[2]*phi_z/c ;

#elif defined WARPX_DIM_XZ

                    const amrex::Real phi_z = - (Eym(i+1,j,0) - Eym(i,j,0))/dx[0] ;
                    Bz(i, j, 0) = Bz_tmp(i,j,0) + dx[2]*phi_z/c ;
                    amrex::ignore_unused(k);
#endif
                }
                );
/*
            // Overwrite the values at time step n+1/2 kept in the temporary arrays on the regular arrays
            // For E
            amrex::ParallelFor(
                tex, tey, tez,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if defined WARPX_DIM_3D
                    Ex(i,j,k) = Ex_tmp(i,j,k);

#elif defined WARPX_DIM_XZ
                    Ex(i,j,0) = Ex_tmp(i,j,0);
                    amrex::ignore_unused(k);

#endif
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if defined WARPX_DIM_3D
                    Eyh(i,j,k) = Eyh_tmp(i,j,k);

#elif defined WARPX_DIM_XZ
                    Ey(i,j,0) = Ey_tmp(i,j,0);
                    amrex::ignore_unused(k);
#endif
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if defined WARPX_DIM_3D
                    Ezh(i,j,k) = Ezh_tmp(i,j,k);

#elif defined WARPX_DIM_XZ
                    Ez(i,j,0) = Ez_tmp(i,j,0);
                    amrex::ignore_unused(k);

#endif
                }
                );

            // For B
            amrex::ParallelFor(
                tbx, tby, tbz,
                [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if defined WARPX_DIM_3D
                    Bxh(i,j,k) = Bxh_tmp(i,j,k)/c;

#elif defined WARPX_DIM_XZ
                    Bx(i,j,0) = Bx_tmp(i,j,0)/c;
                    amrex::ignore_unused(k);
#endif
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if defined WARPX_DIM_3D
                    Byh(i,j,k) = Byh_tmp(i,j,k)/c;

#elif defined WARPX_DIM_XZ
                    By(i,j,0) = By_tmp(i,j,0)/c;
                    amrex::ignore_unused(k);

#endif
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k){

#if defined WARPX_DIM_3D
                    Bz(i,j,k) = Bz_tmp(i,j,k)/c;

#elif defined WARPX_DIM_XZ
                    Bz(i,j,0) = Bz_tmp(i,j,0)/c;
                    amrex::ignore_unused(k);

#endif
                }
                );
        */
        }
        /*
        amrex::MultiFab::Copy(*Efield[0], mf_Ex_tmp, 0, 0, 1, 0);
        amrex::MultiFab::Copy(*Efield[1], mf_Ey_tmp, 0, 0, 1, 0);
        amrex::MultiFab::Copy(*Efield[2], mf_Ez_tmp, 0, 0, 1, 0);
        amrex::MultiFab::Copy(*Bfield[0], mf_Bx_tmp, 0, 0, 1, 0);
        amrex::MultiFab::Copy(*Bfield[1], mf_By_tmp, 0, 0, 1, 0);
        amrex::MultiFab::Copy(*Bfield[2], mf_Bz_tmp, 0, 0, 1, 0);
        Bfield[0]->mult(1./c);
        Bfield[1]->mult(1./c);
        Bfield[2]->mult(1./c);
        */
        std::cout << "After Loop MFI for half time step" << std::endl ;
    }
}
