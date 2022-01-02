#ifndef PARALLEL_COMPUTING_TOOLS_MPI_HPP_
#define PARALLEL_COMPUTING_TOOLS_MPI_HPP_

#include <my/mpi/export.h>

#include <mpi.h>

#include <cstddef>
#include <stdexcept>

namespace my::mpi
{

using code_t = int;

class MY_MPI_EXPORT Exception : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
    ~Exception() override;
};

inline void check_code(code_t mpi_code)
{
    if (mpi_code != MPI_SUCCESS)
    {
        throw Exception("");
    }
}

class MY_MPI_EXPORT Control
{
public:
    Control(int argc, char* argv[])
    {
        if (m_count != 0)
        {
            throw std::runtime_error("Only one Control object must be created");
        }
        code_t mpi_code = MPI_Init(&argc, &argv);
        check_code(mpi_code);
        ++m_count;
    }

    ~Control()
    {
        try
        {
            code_t mpi_code = MPI_Finalize();
            check_code(mpi_code);
        }
        catch (...)
        {
            std::exit(EXIT_FAILURE);
        }
    }

private:
    static int m_count;
};

class Params
{
public:
    static Params& get_instance()
    {
        static Params instance;
        return instance;
    }

    Params(const Params&) = delete;
    Params& operator=(const Params&) = delete;
    Params(Params&&) = delete;
    Params& operator=(Params&&) = delete;

    std::size_t process_id() const
    {
        return m_process_id;
    }

    std::size_t process_count() const
    {
        return m_process_count;
    }

private:
    Params()
    {
        code_t mpi_code;
        mpi_code = MPI_Comm_size(MPI_COMM_WORLD, &m_process_count);
        check_code(mpi_code);
        mpi_code = MPI_Comm_rank(MPI_COMM_WORLD, &m_process_id);
        check_code(mpi_code);
    }

    int m_process_id;
    int m_process_count;
};

inline constexpr int ROOT_ID = 0;
inline constexpr int TAG = 0;

// Cannot mark as inline because MPI_COMM_WORLD is not
// a constant expression in some MPI implementations (e.g. OpenMPI)
MY_MPI_EXPORT extern const MPI_Comm COMM;

inline bool is_current_process_root()
{
    return Params::get_instance().process_id() == ROOT_ID;
}

inline void recv(void* buf, int count, MPI_Datatype datatype, int source)
{
    MPI_Status status;
    code_t code = MPI_Recv(buf, count, datatype, source, TAG, COMM, &status);
    check_code(code);
}

inline void send(const void* buf, int count, MPI_Datatype datatype, int dest)
{
    code_t code = MPI_Send(buf, count, datatype, dest, TAG, COMM);
    check_code(code);
}

inline void bcast(void* buffer, int count, MPI_Datatype datatype)
{
    code_t code = MPI_Bcast(buffer, count, datatype, ROOT_ID, COMM);
    check_code(code);
}

inline void scatterv(const void* sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype,
                     void* recvbuf, int recvcount, MPI_Datatype recvtype)
{
    code_t code = MPI_Scatterv(sendbuf, sendcounts, displs, sendtype,
                               recvbuf, recvcount, recvtype, ROOT_ID, COMM);
    check_code(code);
}

inline void reduce(const void* sendbuf, void* recvbuf, int count,
                   MPI_Datatype datatype, MPI_Op op)
{
    code_t code = MPI_Reduce(sendbuf, recvbuf, count, datatype, op, ROOT_ID, COMM);
    check_code(code);
}

}  // namespace my::mpi

#endif  // PARALLEL_COMPUTING_TOOLS_MPI_HPP_
