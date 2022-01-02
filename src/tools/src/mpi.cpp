#include <mpi.hpp>

namespace my::mpi
{

Exception::~Exception() = default;

int Control::m_count = 0;

extern const MPI_Comm COMM = MPI_COMM_WORLD;

}  // namespace myLLmpi
