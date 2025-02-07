/* Copyright 2019-2020 Andrew Myers, Axel Huebl, Maxence Thevenet,
 * Revathi Jambunathan, Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "InjectorMomentum.H"
#include "PlasmaInjector.H"


using namespace amrex;

void InjectorMomentum::clear ()
{
    switch (type)
    {
    case Type::parser:
    {
        object.parser.m_ux_parser.clear();
        object.parser.m_uy_parser.clear();
        object.parser.m_uz_parser.clear();
        break;
    }
    case Type::gaussian:
    case Type::gaussianflux:
    case Type::boltzmann:
    case Type::juttner:
    case Type::constant:
    case Type::radial_expansion:
    {
        break;
    }
    case Type::custom:
    {
        object.custom.clear();
        break;
    }
    }
}
