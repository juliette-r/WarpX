/* Copyright 2019 Maxence Thevenet, Remi Lehe, Revathi Jambunathan
 *
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "SpectralFieldData.H"
#include "WarpX.H"

#include <map>

#if WARPX_USE_PSATD

using namespace amrex;

/* \brief Initialize fields in spectral space, and FFT plans */
SpectralFieldData::SpectralFieldData( const int lev,
                                      const amrex::BoxArray& realspace_ba,
                                      const SpectralKSpace& k_space,
                                      const amrex::DistributionMapping& dm,
                                      const int n_field_required,
                                      const bool periodic_single_box)
{
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    m_periodic_single_box = periodic_single_box;

    const BoxArray& spectralspace_ba = k_space.spectralspace_ba;

    // Allocate the arrays that contain the fields in spectral space
    // (one component per field)
    fields = SpectralField(spectralspace_ba, dm, n_field_required, 0);

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    tmpRealField = MultiFab(realspace_ba, dm, 1, 0);
    tmpSpectralField = SpectralField(spectralspace_ba, dm, 1, 0);

    // By default, we assume the FFT is done from/to a nodal grid in real space
    // It the FFT is performed from/to a cell-centered grid in real space,
    // a correcting "shift" factor must be applied in spectral space.
    xshift_FFTfromCell = k_space.getSpectralShiftFactor(dm, 0,
                                    ShiftType::TransformFromCellCentered);
    xshift_FFTtoCell = k_space.getSpectralShiftFactor(dm, 0,
                                    ShiftType::TransformToCellCentered);
#if (AMREX_SPACEDIM == 3)
    yshift_FFTfromCell = k_space.getSpectralShiftFactor(dm, 1,
                                    ShiftType::TransformFromCellCentered);
    yshift_FFTtoCell = k_space.getSpectralShiftFactor(dm, 1,
                                    ShiftType::TransformToCellCentered);
    zshift_FFTfromCell = k_space.getSpectralShiftFactor(dm, 2,
                                    ShiftType::TransformFromCellCentered);
    zshift_FFTtoCell = k_space.getSpectralShiftFactor(dm, 2,
                                    ShiftType::TransformToCellCentered);
#else
    zshift_FFTfromCell = k_space.getSpectralShiftFactor(dm, 1,
                                    ShiftType::TransformFromCellCentered);
    zshift_FFTtoCell = k_space.getSpectralShiftFactor(dm, 1,
                                    ShiftType::TransformToCellCentered);
#endif

    // Allocate and initialize the FFT plans
    forward_plan = AnyFFT::FFTplans(spectralspace_ba, dm);
    backward_plan = AnyFFT::FFTplans(spectralspace_ba, dm);
    // Loop over boxes and allocate the corresponding plan
    // for each box owned by the local MPI proc
    for ( MFIter mfi(spectralspace_ba, dm); mfi.isValid(); ++mfi ){
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = amrex::second();

        // Note: the size of the real-space box and spectral-space box
        // differ when using real-to-complex FFT. When initializing
        // the FFT plan, the valid dimensions are those of the real-space box.
        IntVect fft_size = realspace_ba[mfi].length();

        forward_plan[mfi] = AnyFFT::CreatePlan(
            fft_size, tmpRealField[mfi].dataPtr(),
            reinterpret_cast<AnyFFT::Complex*>( tmpSpectralField[mfi].dataPtr()),
            AnyFFT::direction::R2C, AMREX_SPACEDIM);

        backward_plan[mfi] = AnyFFT::CreatePlan(
            fft_size, tmpRealField[mfi].dataPtr(),
            reinterpret_cast<AnyFFT::Complex*>( tmpSpectralField[mfi].dataPtr()),
            AnyFFT::direction::C2R, AMREX_SPACEDIM);

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = amrex::second() - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}


SpectralFieldData::~SpectralFieldData()
{
    if (tmpRealField.size() > 0){
        for ( MFIter mfi(tmpRealField); mfi.isValid(); ++mfi ){
            AnyFFT::DestroyPlan(forward_plan[mfi]);
            AnyFFT::DestroyPlan(backward_plan[mfi]);
        }
    }
}

/* \brief Transform the component `i_comp` of MultiFab `mf`
 *  to spectral space, and store the corresponding result internally
 *  (in the spectral field specified by `field_index`) */
void
SpectralFieldData::ForwardTransform (const int lev,
                     const MultiFab& mf, const int field_index,
                                     const int i_comp, const IntVect& stag)
{
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Check field index type, in order to apply proper shift in spectral space
    const bool is_nodal_x = (stag[0] == amrex::IndexType::NODE) ? true : false;
#if (AMREX_SPACEDIM == 3)
    const bool is_nodal_y = (stag[1] == amrex::IndexType::NODE) ? true : false;
    const bool is_nodal_z = (stag[2] == amrex::IndexType::NODE) ? true : false;
#else
    const bool is_nodal_z = (stag[1] == amrex::IndexType::NODE) ? true : false;
#endif

    // Loop over boxes
    for ( MFIter mfi(mf); mfi.isValid(); ++mfi ){
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = amrex::second();

        // Copy the real-space field `mf` to the temporary field `tmpRealField`
        // This ensures that all fields have the same number of points
        // before the Fourier transform.
        // As a consequence, the copy discards the *last* point of `mf`
        // in any direction that has *nodal* index type.
        {
            Box realspace_bx;
            if (m_periodic_single_box) {
                realspace_bx = mfi.validbox(); // Discard guard cells
            } else {
                realspace_bx = mf[mfi].box(); // Keep guard cells
            }
            realspace_bx.enclosedCells(); // Discard last point in nodal direction
            AMREX_ALWAYS_ASSERT( realspace_bx.contains(tmpRealField[mfi].box()) );
            Array4<const Real> mf_arr = mf[mfi].array();
            Array4<Real> tmp_arr = tmpRealField[mfi].array();
            ParallelFor( tmpRealField[mfi].box(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                tmp_arr(i,j,k) = mf_arr(i,j,k,i_comp);
            });
        }

        // Perform Fourier transform from `tmpRealField` to `tmpSpectralField`
        AnyFFT::Execute(forward_plan[mfi]);

        // Copy the spectral-space field `tmpSpectralField` to the appropriate
        // index of the FabArray `fields` (specified by `field_index`)
        // and apply correcting shift factor if the real space data comes
        // from a cell-centered grid in real space instead of a nodal grid.
        {
            Array4<Complex> fields_arr = SpectralFieldData::fields[mfi].array();
            Array4<const Complex> tmp_arr = tmpSpectralField[mfi].array();
            const Complex* xshift_arr = xshift_FFTfromCell[mfi].dataPtr();
#if (AMREX_SPACEDIM == 3)
            const Complex* yshift_arr = yshift_FFTfromCell[mfi].dataPtr();
#endif
            const Complex* zshift_arr = zshift_FFTfromCell[mfi].dataPtr();
            // Loop over indices within one box
            const Box spectralspace_bx = tmpSpectralField[mfi].box();

            ParallelFor( spectralspace_bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                Complex spectral_field_value = tmp_arr(i,j,k);
                // Apply proper shift in each dimension
                if (is_nodal_x==false) spectral_field_value *= xshift_arr[i];
#if (AMREX_SPACEDIM == 3)
                if (is_nodal_y==false) spectral_field_value *= yshift_arr[j];
                if (is_nodal_z==false) spectral_field_value *= zshift_arr[k];
#elif (AMREX_SPACEDIM == 2)
                if (is_nodal_z==false) spectral_field_value *= zshift_arr[j];
#endif
                // Copy field into the right index
                fields_arr(i,j,k,field_index) = spectral_field_value;
            });
        }

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = amrex::second() - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}


/* \brief Transform spectral field specified by `field_index` back to
 * real space, and store it in the component `i_comp` of `mf` */
void
SpectralFieldData::BackwardTransform( const int lev,
                                      MultiFab& mf,
                                      const int field_index,
                                      const int i_comp )
{
    amrex::LayoutData<amrex::Real>* cost = WarpX::getCosts(lev);

    // Check field index type, in order to apply proper shift in spectral space
    const bool is_nodal_x = mf.is_nodal(0);
#if (AMREX_SPACEDIM == 3)
    const bool is_nodal_y = mf.is_nodal(1);
    const bool is_nodal_z = mf.is_nodal(2);
#else
    const bool is_nodal_z = mf.is_nodal(1);
#endif

    const int si = (is_nodal_x) ? 1 : 0;
#if   (AMREX_SPACEDIM == 2)
    const int sj = (is_nodal_z) ? 1 : 0;
    const int sk = 0;
#elif (AMREX_SPACEDIM == 3)
    const int sj = (is_nodal_y) ? 1 : 0;
    const int sk = (is_nodal_z) ? 1 : 0;
#endif

    // Loop over boxes
    for ( MFIter mfi(mf); mfi.isValid(); ++mfi ){
        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        Real wt = amrex::second();

        // Copy the spectral-space field `tmpSpectralField` to the appropriate
        // field (specified by the input argument field_index)
        // and apply correcting shift factor if the field is to be transformed
        // to a cell-centered grid in real space instead of a nodal grid.
        {
            Array4<const Complex> field_arr = SpectralFieldData::fields[mfi].array();
            Array4<Complex> tmp_arr = tmpSpectralField[mfi].array();
            const Complex* xshift_arr = xshift_FFTtoCell[mfi].dataPtr();
#if (AMREX_SPACEDIM == 3)
            const Complex* yshift_arr = yshift_FFTtoCell[mfi].dataPtr();
#endif
            const Complex* zshift_arr = zshift_FFTtoCell[mfi].dataPtr();
            // Loop over indices within one box
            const Box spectralspace_bx = tmpSpectralField[mfi].box();

            ParallelFor( spectralspace_bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                Complex spectral_field_value = field_arr(i,j,k,field_index);
                // Apply proper shift in each dimension
                if (is_nodal_x==false) spectral_field_value *= xshift_arr[i];
#if (AMREX_SPACEDIM == 3)
                if (is_nodal_y==false) spectral_field_value *= yshift_arr[j];
                if (is_nodal_z==false) spectral_field_value *= zshift_arr[k];
#elif (AMREX_SPACEDIM == 2)
                if (is_nodal_z==false) spectral_field_value *= zshift_arr[j];
#endif
                // Copy field into temporary array
                tmp_arr(i,j,k) = spectral_field_value;
            });
        }

        // Perform Fourier transform from `tmpSpectralField` to `tmpRealField`
        AnyFFT::Execute(backward_plan[mfi]);

        // Copy the temporary field tmpRealField to the real-space field mf and
        // normalize, dividing by N, since (FFT + inverse FFT) results in a factor N
        {
            amrex::Box const& mf_box = (m_periodic_single_box) ? mfi.validbox() : mfi.fabbox();
            amrex::Array4<amrex::Real> mf_arr = mf[mfi].array();
            amrex::Array4<const amrex::Real> tmp_arr = tmpRealField[mfi].array();

            const amrex::Real inv_N = 1._rt / tmpRealField[mfi].box().numPts();

            // Total number of cells, including ghost cells (nj represents ny in 3D and nz in 2D)
            const int ni = mf_box.length(0);
            const int nj = mf_box.length(1);
#if   (AMREX_SPACEDIM == 2)
            constexpr int nk = 1;
#elif (AMREX_SPACEDIM == 3)
            const int nk = mf_box.length(2);
#endif
            // Lower bound of the box (lo_j represents lo_y in 3D and lo_z in 2D)
            const int lo_i = amrex::lbound(mf_box).x;
            const int lo_j = amrex::lbound(mf_box).y;
#if   (AMREX_SPACEDIM == 2)
            constexpr int lo_k = 0;
#elif (AMREX_SPACEDIM == 3)
            const int lo_k = amrex::lbound(mf_box).z;
#endif
            // Loop over cells within full box, including ghost cells
            ParallelFor(mf_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                // Assume periodicity and set the last outer guard cell equal to the first one:
                // this is necessary in order to get the correct value along a nodal direction,
                // because the last point along a nodal direction is always discarded when FFTs
                // are computed, as the real-space box is always cell-centered.
                const int ii = (i == lo_i + ni - si) ? lo_i : i;
                const int jj = (j == lo_j + nj - sj) ? lo_j : j;
                const int kk = (k == lo_k + nk - sk) ? lo_k : k;
                // Copy and normalize field
                mf_arr(i,j,k,i_comp) = inv_N * tmp_arr(ii,jj,kk);
            });
        }

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = amrex::second() - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[mfi.index()], wt);
        }
    }
}

#endif // WARPX_USE_PSATD
