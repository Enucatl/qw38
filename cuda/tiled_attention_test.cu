#include "attention_decode.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#include <vector>

namespace {
using qw38::cuda::AttentionCache;
using qw38::cuda::AttentionConfig;
struct B { float *q{},*k{},*v{},*g{},*qs{},*ks{},*nq{},*nk{},*score{},*out{}; __nv_bfloat16 *ck{},*cv{},*tk{},*tv{}; };
void release(B& b){cudaFree(b.tv);cudaFree(b.tk);cudaFree(b.cv);cudaFree(b.ck);cudaFree(b.out);cudaFree(b.score);cudaFree(b.nk);cudaFree(b.nq);cudaFree(b.ks);cudaFree(b.qs);cudaFree(b.g);cudaFree(b.v);cudaFree(b.k);cudaFree(b.q);}
bool allocate(B& b,const AttentionConfig& c,size_t rows,size_t start){size_t q=qw38::cuda::attention_query_values(c),r=qw38::cuda::attention_kv_row_values(c),cv=qw38::cuda::attention_cache_values(c),s=qw38::cuda::attention_chunk_score_values(c,start,rows);
#define A(x,n) if(cudaMalloc(reinterpret_cast<void**>(&b.x),(n)*sizeof(*b.x))!=cudaSuccess)return false
A(q,rows*q);A(k,rows*r);A(v,rows*r);A(g,rows*q);A(qs,c.head_width);A(ks,c.head_width);A(nq,q);A(nk,r);A(score,s);A(out,rows*q);A(ck,cv);A(cv,cv);A(tk,rows*r);A(tv,rows*r);
#undef A
return true;}
void seed(B& b,const AttentionConfig& c,size_t rows,size_t start){size_t q=qw38::cuda::attention_query_values(c),r=qw38::cuda::attention_kv_row_values(c),cv=qw38::cuda::attention_cache_values(c);std::vector<float> hq(rows*q),hk(rows*r),hv(rows*r),hg(rows*q),s(c.head_width,1.f);for(size_t i=0;i<hq.size();++i){hq[i]=sinf(float(i)*.001f);hg[i]=cosf(float(i)*.002f);}for(size_t i=0;i<hk.size();++i){hk[i]=cosf(float(i)*.003f);hv[i]=sinf(float(i)*.004f);}std::vector<__nv_bfloat16> z(cv);for(size_t i=0;i<cv;++i)z[i]=__float2bfloat16_rn(float(int(i%31)-15)*.015625f);std::vector<__nv_bfloat16> physical(cv);qw38::cuda::attention_kv_copy_logical_to_physical(z.data(),physical.data(),c.kv_heads,c.capacity,c.head_width);cudaMemcpy(b.q,hq.data(),hq.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b.k,hk.data(),hk.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b.v,hv.data(),hv.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b.g,hg.data(),hg.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b.qs,s.data(),s.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b.ks,s.data(),s.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(b.ck,physical.data(),physical.size()*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice);cudaMemset(b.score,0xA5,qw38::cuda::attention_chunk_score_values(c,start,rows)*sizeof(float));}
void gather_logical_rows(const __nv_bfloat16* device,std::vector<__nv_bfloat16>* logical,size_t start,size_t rows,const AttentionConfig& c){std::vector<__nv_bfloat16> physical(qw38::cuda::attention_cache_values(c));cudaMemcpy(physical.data(),device,physical.size()*sizeof(physical[0]),cudaMemcpyDeviceToHost);logical->assign(rows*qw38::cuda::attention_kv_row_values(c),__nv_bfloat16{});qw38::cuda::attention_kv_gather_logical_rows(physical.data(),logical->data(),start,rows,c.kv_heads,c.capacity,c.head_width);}
void scatter_logical_row(__nv_bfloat16* device,const std::vector<__nv_bfloat16>& row,size_t token,const AttentionConfig& c){for(std::uint32_t head=0;head<c.kv_heads;++head)cudaMemcpy(device+qw38::cuda::attention_kv_physical_index(token,head,0,c.capacity,c.head_width),row.data()+head*c.head_width,c.head_width*sizeof(row[0]),cudaMemcpyHostToDevice);}
cudaError_t invoke(const AttentionConfig& c,size_t start,size_t rows,B& b,bool ref,cudaStream_t stream){AttentionCache committed{b.ck,b.cv},candidate{b.tk,b.tv};return ref?qw38::cuda::launch_attention_prepare_chunk_reference(c,start,rows,b.q,b.k,b.v,b.qs,b.ks,b.g,committed,candidate,b.nq,b.nk,b.score,b.out,stream):qw38::cuda::launch_attention_prepare_chunk(c,start,rows,b.q,b.k,b.v,b.qs,b.ks,b.g,committed,candidate,b.nq,b.nk,b.score,b.out,stream);}
int capture(const AttentionConfig& c,size_t start,size_t rows,B& b,bool ref){cudaStream_t s{};cudaGraph_t g{};cudaGraphExec_t e{};if(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking)!=cudaSuccess)return -1;if(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal)!=cudaSuccess)return -1;cudaError_t x=invoke(c,start,rows,b,ref,s);if(x==cudaSuccess)x=cudaStreamEndCapture(s,&g);if(x!=cudaSuccess){cudaStreamDestroy(s);return -1;}size_t n=0;std::vector<cudaGraphNode_t> nodes;if(x==cudaSuccess)x=cudaGraphGetNodes(g,nullptr,&n);if(x==cudaSuccess){nodes.resize(n);x=cudaGraphGetNodes(g,nodes.data(),&n);}if(x==cudaSuccess)x=cudaGraphInstantiate(&e,g,nullptr,nullptr,0);if(x==cudaSuccess)x=cudaGraphLaunch(e,s);if(x==cudaSuccess)x=cudaStreamSynchronize(s);int k=0;for(auto node:nodes){cudaGraphNodeType t{};if(cudaGraphNodeGetType(node,&t)==cudaSuccess&&t==cudaGraphNodeTypeKernel)++k;}cudaGraphExecDestroy(e);cudaGraphDestroy(g);cudaStreamDestroy(s);return x==cudaSuccess?k:-1;}
float maxabs(const std::vector<float>&a,const std::vector<float>&b){float m=0;for(size_t i=0;i<a.size();++i)m=fmaxf(m,fabsf(a[i]-b[i]));return m;}float rms(const std::vector<float>&a,const std::vector<float>&b){double s=0;for(size_t i=0;i<a.size();++i){double d=a[i]-b[i];s+=d*d;}return sqrtf(float(s/a.size()));}float cosine(const std::vector<float>&a,const std::vector<float>&b){double ab=0,aa=0,bb=0;for(size_t i=0;i<a.size();++i){ab+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}return float(ab/sqrt(aa*bb));}
bool envelope_ok(const std::vector<float>& a, const std::vector<float>& b) {
  return maxabs(a, b) <= 5e-5f && rms(a, b) <= 5e-6f;
}
bool finite_vec(const std::vector<float>& values) {
  for (float value : values)
    if (!std::isfinite(value)) return false;
  return true;
}
cudaError_t invoke_tiled(const AttentionConfig& c, size_t start, size_t rows,
                         B& b) {
  AttentionCache committed{b.ck, b.cv}, candidate{b.tk, b.tv};
  return qw38::cuda::launch_attention_prepare_chunk_tiled(
      c, start, rows, b.q, b.k, b.v, b.qs, b.ks, b.g, committed, candidate,
      b.nq, b.nk, b.score, b.out, nullptr);
}
cudaError_t invoke_rank3(const AttentionConfig& c, size_t start, size_t rows,
                         B& b) {
  AttentionCache committed{b.ck, b.cv}, candidate{b.tk, b.tv};
  return qw38::cuda::launch_attention_prepare_chunk_mma_rank3(
      c, start, rows, b.q, b.k, b.v, b.qs, b.ks, b.g, committed, candidate,
      b.nq, b.nk, b.score, b.out, nullptr);
}
cudaError_t invoke_ncols1(const AttentionConfig& c, size_t start, size_t rows,
                          B& b, int ncols1) {
  AttentionCache committed{b.ck, b.cv}, candidate{b.tk, b.tv};
  return qw38::cuda::launch_attention_prepare_chunk_mma_ncols1(
      c, start, rows, b.q, b.k, b.v, b.qs, b.ks, b.g, committed, candidate,
      b.nq, b.nk, b.score, b.out, ncols1, nullptr);
}
}
int main(){
  AttentionConfig c{24,4,256,64,131072}; size_t rows=9,start=5,q=qw38::cuda::attention_query_values(c),kv=qw38::cuda::attention_kv_row_values(c),cache=qw38::cuda::attention_cache_values(c),s=qw38::cuda::attention_chunk_score_values(c,start,rows); B t{},r{},repeated{},future{},later{},commit_check{};
  if(!allocate(t,c,rows,start)||!allocate(r,c,rows,start)||!allocate(repeated,c,rows,start)||!allocate(future,c,rows,start)||!allocate(later,c,rows,start)||!allocate(commit_check,c,rows,start)) return 2; seed(t,c,rows,start); seed(r,c,rows,start); seed(repeated,c,rows,start); seed(future,c,rows,start); seed(later,c,rows,start);seed(commit_check,c,rows,start);
  std::vector<float>a(rows*q),b(a.size()),before(s),after(s); std::vector<__nv_bfloat16> cache_key_before(cache),cache_value_before(cache),cache_key_after(cache),cache_value_after(cache); cudaMemcpy(before.data(),t.score,s*sizeof(float),cudaMemcpyDeviceToHost);cudaMemcpy(cache_key_before.data(),t.ck,cache*sizeof(cache_key_before[0]),cudaMemcpyDeviceToHost);cudaMemcpy(cache_value_before.data(),t.cv,cache*sizeof(cache_value_before[0]),cudaMemcpyDeviceToHost);
  if(invoke(c,start,rows,t,false,nullptr)!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)return 3;
  if(invoke(c,start,rows,r,true,nullptr)!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)return 4;
  cudaMemcpy(a.data(),t.out,a.size()*sizeof(float),cudaMemcpyDeviceToHost); cudaMemcpy(b.data(),r.out,b.size()*sizeof(float),cudaMemcpyDeviceToHost); cudaMemcpy(after.data(),t.score,s*sizeof(float),cudaMemcpyDeviceToHost);
  bool scratch=memcmp(before.data(),after.data(),s*sizeof(float))==0,finite=true; for(float v:a)finite&=std::isfinite(v); float ma=maxabs(a,b),rr=rms(a,b),co=cosine(a,b);
  std::vector<__nv_bfloat16> tc(rows*kv),tv(rows*kv),rc(rows*kv),rv(rows*kv); cudaMemcpy(tc.data(),t.tk,tc.size()*sizeof(tc[0]),cudaMemcpyDeviceToHost);cudaMemcpy(tv.data(),t.tv,tv.size()*sizeof(tv[0]),cudaMemcpyDeviceToHost);cudaMemcpy(rc.data(),r.tk,rc.size()*sizeof(rc[0]),cudaMemcpyDeviceToHost);cudaMemcpy(rv.data(),r.tv,rv.size()*sizeof(rv[0]),cudaMemcpyDeviceToHost);cudaMemcpy(cache_key_after.data(),t.ck,cache*sizeof(cache_key_after[0]),cudaMemcpyDeviceToHost);cudaMemcpy(cache_value_after.data(),t.cv,cache*sizeof(cache_value_after[0]),cudaMemcpyDeviceToHost);
  bool candidate_exact=memcmp(tc.data(),rc.data(),tc.size()*sizeof(tc[0]))==0&&memcmp(tv.data(),rv.data(),tv.size()*sizeof(tv[0]))==0; bool cache_unchanged=memcmp(cache_key_before.data(),cache_key_after.data(),cache*sizeof(cache_key_before[0]))==0&&memcmp(cache_value_before.data(),cache_value_after.data(),cache*sizeof(cache_value_before[0]))==0;
  bool zero_rejected=invoke(c,start,0,t,false,nullptr)==cudaErrorInvalidValue; bool overflow_rejected=invoke(c,c.capacity,1,t,false,nullptr)==cudaErrorInvalidValue; bool alias_rejected=qw38::cuda::launch_attention_prepare_chunk(c,start,1,t.q,t.k,t.v,t.qs,t.ks,t.g,{t.ck,t.cv},{t.ck,t.tv},t.nq,t.nk,t.score,t.out,nullptr)==cudaErrorInvalidValue;
  float ma3=0,rr3=0,co3=0; if(invoke(c,start,3,t,false,nullptr)!=cudaSuccess||invoke(c,start,3,r,true,nullptr)!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)return 6;std::vector<float>a3(3*q),b3(3*q);cudaMemcpy(a3.data(),t.out,a3.size()*sizeof(float),cudaMemcpyDeviceToHost);cudaMemcpy(b3.data(),r.out,b3.size()*sizeof(float),cudaMemcpyDeviceToHost);ma3=maxabs(a3,b3);rr3=rms(a3,b3);co3=cosine(a3,b3);
  bool normalized_equal=true; if(invoke(c,start,1,t,false,nullptr)!=cudaSuccess||invoke(c,start,1,r,true,nullptr)!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)return 6;std::vector<float>nqt(q),nqr(q),nkt(kv),nkr(kv);cudaMemcpy(nqt.data(),t.nq,q*sizeof(float),cudaMemcpyDeviceToHost);cudaMemcpy(nqr.data(),r.nq,q*sizeof(float),cudaMemcpyDeviceToHost);cudaMemcpy(nkt.data(),t.nk,kv*sizeof(float),cudaMemcpyDeviceToHost);cudaMemcpy(nkr.data(),r.nk,kv*sizeof(float),cudaMemcpyDeviceToHost);normalized_equal=memcmp(nqt.data(),nqr.data(),q*sizeof(float))==0&&memcmp(nkt.data(),nkr.data(),kv*sizeof(float))==0;
  bool last_position=invoke(c,c.capacity-1,1,t,false,nullptr)==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess;
  std::uint64_t frontier_value=17,*frontier{};cudaMalloc(reinterpret_cast<void**>(&frontier),sizeof(frontier_value));cudaMemcpy(frontier,&frontier_value,sizeof(frontier_value),cudaMemcpyHostToDevice);std::uint64_t frontier_after=0;cudaMemcpy(&frontier_after,frontier,sizeof(frontier_after),cudaMemcpyDeviceToHost);bool frontier_unchanged=frontier_after==frontier_value;
  std::uint64_t repeated_frontier_value=start,*repeated_frontier{};cudaMalloc(reinterpret_cast<void**>(&repeated_frontier),sizeof(repeated_frontier_value));cudaMemcpy(repeated_frontier,&repeated_frontier_value,sizeof(repeated_frontier_value),cudaMemcpyHostToDevice);bool repeated_ok=true;for(size_t i=0;i<rows;++i){AttentionCache committed_cache{repeated.ck,repeated.cv},candidate_cache{repeated.tk+i*kv,repeated.tv+i*kv};repeated_ok&=qw38::cuda::launch_attention_prepare_chunk(c,start+i,1,repeated.q+i*q,repeated.k+i*kv,repeated.v+i*kv,repeated.qs,repeated.ks,repeated.g+i*q,committed_cache,candidate_cache,repeated.nq,repeated.nk,repeated.score,repeated.out+i*q,nullptr)==cudaSuccess;repeated_ok&=cudaDeviceSynchronize()==cudaSuccess;repeated_ok&=qw38::cuda::launch_attention_commit_chunk(c,start+i,1,candidate_cache,committed_cache,start+i+1,repeated_frontier,nullptr)==cudaSuccess;repeated_ok&=cudaDeviceSynchronize()==cudaSuccess;}std::vector<float>repeated_output(rows*q);std::vector<__nv_bfloat16>repeated_key(rows*kv),repeated_value(rows*kv);std::uint64_t repeated_frontier_after=0;cudaMemcpy(repeated_output.data(),repeated.out,repeated_output.size()*sizeof(float),cudaMemcpyDeviceToHost);gather_logical_rows(repeated.ck,&repeated_key,start,rows,c);gather_logical_rows(repeated.cv,&repeated_value,start,rows,c);cudaMemcpy(&repeated_frontier_after,repeated_frontier,sizeof(repeated_frontier_after),cudaMemcpyDeviceToHost);bool chunk_output=repeated_ok&&memcmp(a.data(),repeated_output.data(),a.size()*sizeof(float))==0;bool chunk_candidate=repeated_ok&&memcmp(tc.data(),repeated_key.data(),tc.size()*sizeof(tc[0]))==0&&memcmp(tv.data(),repeated_value.data(),tv.size()*sizeof(tv[0]))==0&&repeated_frontier_after==start+rows;cudaFree(repeated_frontier);
  std::vector<__nv_bfloat16> sentinel_row(kv,__float2bfloat16_rn(123.f));scatter_logical_row(future.ck,sentinel_row,start+rows+1,c);scatter_logical_row(future.cv,sentinel_row,start+rows+1,c);bool future_ok=invoke(c,start,rows,future,false,nullptr)==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess;std::vector<float>future_output(rows*q);cudaMemcpy(future_output.data(),future.out,future_output.size()*sizeof(float),cudaMemcpyDeviceToHost);bool future_excluded=future_ok&&memcmp(a.data(),future_output.data(),a.size()*sizeof(float))==0;
  std::vector<float>later_key(kv,321.f),later_value(kv,-321.f);cudaMemcpy(later.k+(rows-1)*kv,later_key.data(),later_key.size()*sizeof(float),cudaMemcpyHostToDevice);cudaMemcpy(later.v+(rows-1)*kv,later_value.data(),later_value.size()*sizeof(float),cudaMemcpyHostToDevice);bool later_ok=invoke(c,start,rows,later,false,nullptr)==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess;std::vector<float>later_output(rows*q);cudaMemcpy(later_output.data(),later.out,later_output.size()*sizeof(float),cudaMemcpyDeviceToHost);bool later_excluded=later_ok&&memcmp(a.data(),later_output.data(),(rows-1)*q*sizeof(float))==0;
  bool commit_prepared=invoke(c,start,rows,commit_check,false,nullptr)==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess;std::vector<__nv_bfloat16>commit_candidate_key(rows*kv),commit_candidate_value(rows*kv);cudaMemcpy(commit_candidate_key.data(),commit_check.tk,commit_candidate_key.size()*sizeof(commit_candidate_key[0]),cudaMemcpyDeviceToHost);cudaMemcpy(commit_candidate_value.data(),commit_check.tv,commit_candidate_value.size()*sizeof(commit_candidate_value[0]),cudaMemcpyDeviceToHost);bool commit_ok=commit_prepared&&qw38::cuda::launch_attention_commit_chunk(c,start,rows,{commit_check.tk,commit_check.tv},{commit_check.ck,commit_check.cv},start+rows,frontier,nullptr)==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess;cudaMemcpy(&frontier_after,frontier,sizeof(frontier_after),cudaMemcpyDeviceToHost);std::vector<__nv_bfloat16> committed_key(rows*kv),committed_value(rows*kv);gather_logical_rows(commit_check.ck,&committed_key,start,rows,c);gather_logical_rows(commit_check.cv,&committed_value,start,rows,c);bool commit_exact=commit_ok&&memcmp(committed_key.data(),commit_candidate_key.data(),committed_key.size()*sizeof(committed_key[0]))==0&&memcmp(committed_value.data(),commit_candidate_value.data(),committed_value.size()*sizeof(committed_value[0]))==0;bool commit_frontier=frontier_after==start+rows;cudaFree(frontier);
  release(t);release(r);release(repeated);release(future);release(later);release(commit_check);
  B mma64{},tiled64{};if(!allocate(mma64,c,64,start)||!allocate(tiled64,c,64,start))return 5;seed(mma64,c,64,start);seed(tiled64,c,64,start);cudaError_t mma_err=invoke(c,start,64,mma64,false,nullptr);cudaError_t mma_sync=cudaDeviceSynchronize();if(mma_err!=cudaSuccess||mma_sync!=cudaSuccess){fprintf(stderr,"mma 64-row launch failed err=%s sync=%s\n",cudaGetErrorString(mma_err),cudaGetErrorString(mma_sync));return 6;}AttentionCache tiled_committed{tiled64.ck,tiled64.cv},tiled_candidate{tiled64.tk,tiled64.tv};cudaError_t tiled_err=qw38::cuda::launch_attention_prepare_chunk_tiled(c,start,64,tiled64.q,tiled64.k,tiled64.v,tiled64.qs,tiled64.ks,tiled64.g,tiled_committed,tiled_candidate,tiled64.nq,tiled64.nk,tiled64.score,tiled64.out,nullptr);cudaError_t tiled_sync=cudaDeviceSynchronize();if(tiled_err!=cudaSuccess||tiled_sync!=cudaSuccess){fprintf(stderr,"tiled 64-row launch failed err=%s sync=%s\n",cudaGetErrorString(tiled_err),cudaGetErrorString(tiled_sync));return 6;}std::vector<float>mma_out(64*q),tiled_out(64*q);cudaMemcpy(mma_out.data(),mma64.out,mma_out.size()*sizeof(float),cudaMemcpyDeviceToHost);cudaMemcpy(tiled_out.data(),tiled64.out,tiled_out.size()*sizeof(float),cudaMemcpyDeviceToHost);float ma64=maxabs(mma_out,tiled_out),rr64=rms(mma_out,tiled_out);bool mma_envelope=ma64<=5e-5f&&rr64<=5e-6f;release(mma64);release(tiled64);if(!mma_envelope){fprintf(stderr,"mma vs tiled 64-row envelope failed max_abs=%.9g rms=%.9g\n",ma64,rr64);return 6;}
  std::printf("opt019_attention_64 max_abs=%.9g rms=%.9g occupancy=%d shared=%zu selected_ncols1=%d\n",
              ma64, rr64, qw38::cuda::attention_mma_quality_occupancy(),
              qw38::cuda::attention_mma_quality_shared_bytes(),
              qw38::cuda::selected_attention_mma_query_rows());
  {
    const size_t rows2048 = 2048;
    const size_t start2048 = 5;
    B quality{}, tiled{}, rank3{};
    if (!allocate(quality, c, rows2048, start2048) ||
        !allocate(tiled, c, rows2048, start2048) ||
        !allocate(rank3, c, rows2048, start2048))
      return 5;
    seed(quality, c, rows2048, start2048);
    seed(tiled, c, rows2048, start2048);
    seed(rank3, c, rows2048, start2048);
    if (invoke_tiled(c, start2048, rows2048, tiled) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 6;
    std::vector<float> tiled_host(rows2048 * q);
    cudaMemcpy(tiled_host.data(), tiled.out, tiled_host.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    const int legal[3] = {8, 16, 32};
    int winner = 0;
    float winner_ms = 1.0e30f;
    cudaEvent_t e0{}, e1{};
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int ncols1 : legal) {
      seed(quality, c, rows2048, start2048);
      cudaError_t launched =
          invoke_ncols1(c, start2048, rows2048, quality, ncols1);
      cudaError_t synced = cudaDeviceSynchronize();
      std::vector<float> host(rows2048 * q);
      if (launched == cudaSuccess && synced == cudaSuccess) {
        cudaMemcpy(host.data(), quality.out, host.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
      }
      const bool finite = launched == cudaSuccess && synced == cudaSuccess &&
                          finite_vec(host);
      const bool env = finite && envelope_ok(host, tiled_host);
      const int occupancy = qw38::cuda::attention_mma_quality_occupancy_for(ncols1);
      float samples[3] = {0, 0, 0};
      bool timed_ok = env && occupancy >= 1;
      for (int sample = 0; sample < 3 && timed_ok; ++sample) {
        cudaEventRecord(e0);
        timed_ok = invoke_ncols1(c, start2048, rows2048, quality, ncols1) ==
                       cudaSuccess;
        cudaEventRecord(e1);
        timed_ok = timed_ok && cudaEventSynchronize(e1) == cudaSuccess;
        if (timed_ok) cudaEventElapsedTime(&samples[sample], e0, e1);
      }
      const float mean =
          timed_ok ? (samples[0] + samples[1] + samples[2]) / 3.0f : 0.0f;
      std::printf("opt019_attention_ncols1 ncols1=%d launch=%s finite=%s "
                  "envelope=%s occupancy=%d sample0=%.9g sample1=%.9g "
                  "sample2=%.9g mean_ms=%.9g\n",
                  ncols1, launched == cudaSuccess ? "ok" : "fail",
                  finite ? "true" : "false", env ? "true" : "false", occupancy,
                  samples[0], samples[1], samples[2], mean);
      if (timed_ok && mean > 0.0f && mean < winner_ms) {
        winner_ms = mean;
        winner = ncols1;
      }
    }
    seed(quality, c, rows2048, start2048);
    if (invoke(c, start2048, rows2048, quality, false, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 6;
    std::vector<float> quality_host(rows2048 * q);
    cudaMemcpy(quality_host.data(), quality.out,
               quality_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    const float ma2048 = maxabs(quality_host, tiled_host);
    const float rr2048 = rms(quality_host, tiled_host);
    const bool env2048 = envelope_ok(quality_host, tiled_host) &&
                         finite_vec(quality_host);
    std::printf("opt019_attention_2048 max_abs=%.9g rms=%.9g passed=%s "
                "winner_ncols1=%d selected_ncols1=%d\n",
                ma2048, rr2048, env2048 ? "true" : "false", winner,
                qw38::cuda::selected_attention_mma_query_rows());
    float quality_samples[3] = {0, 0, 0};
    float tiled_samples[3] = {0, 0, 0};
    float rank3_samples[3] = {0, 0, 0};
    bool speed_ok = env2048;
    for (int sample = 0; sample < 3 && speed_ok; ++sample) {
      cudaEventRecord(e0);
      speed_ok = invoke(c, start2048, rows2048, quality, false, nullptr) ==
                 cudaSuccess;
      cudaEventRecord(e1);
      speed_ok = speed_ok && cudaEventSynchronize(e1) == cudaSuccess;
      if (speed_ok) cudaEventElapsedTime(&quality_samples[sample], e0, e1);
      cudaEventRecord(e0);
      speed_ok = speed_ok &&
                 invoke_tiled(c, start2048, rows2048, tiled) == cudaSuccess;
      cudaEventRecord(e1);
      speed_ok = speed_ok && cudaEventSynchronize(e1) == cudaSuccess;
      if (speed_ok) cudaEventElapsedTime(&tiled_samples[sample], e0, e1);
      cudaEventRecord(e0);
      speed_ok = speed_ok &&
                 invoke_rank3(c, start2048, rows2048, rank3) == cudaSuccess;
      cudaEventRecord(e1);
      speed_ok = speed_ok && cudaEventSynchronize(e1) == cudaSuccess;
      if (speed_ok) cudaEventElapsedTime(&rank3_samples[sample], e0, e1);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    const float quality_mean =
        (quality_samples[0] + quality_samples[1] + quality_samples[2]) / 3.0f;
    const float tiled_mean =
        (tiled_samples[0] + tiled_samples[1] + tiled_samples[2]) / 3.0f;
    const float rank3_mean =
        (rank3_samples[0] + rank3_samples[1] + rank3_samples[2]) / 3.0f;
    const bool faster = speed_ok && quality_mean > 0.0f &&
                        quality_mean < tiled_mean && quality_mean < rank3_mean &&
                        qw38::cuda::attention_mma_quality_occupancy() >= 1;
    std::printf("opt019_attention_ab quality_ms=%.9g tiled_ms=%.9g rank3_ms=%.9g "
                "faster=%s occupancy=%d samples_q=%.9g,%.9g,%.9g "
                "samples_t=%.9g,%.9g,%.9g samples_r=%.9g,%.9g,%.9g\n",
                quality_mean, tiled_mean, rank3_mean, faster ? "true" : "false",
                qw38::cuda::attention_mma_quality_occupancy(),
                quality_samples[0], quality_samples[1], quality_samples[2],
                tiled_samples[0], tiled_samples[1], tiled_samples[2],
                rank3_samples[0], rank3_samples[1], rank3_samples[2]);
    release(quality);
    release(tiled);
    release(rank3);
    if (!env2048 || !faster || winner == 0) {
      fprintf(stderr,
              "opt019 attention 2048 failed envelope=%d faster=%d winner=%d\n",
              env2048, faster, winner);
      return 6;
    }
  }
  {
    const size_t rows4096 = 4096;
    const size_t start4096 = 0;
    B tiled4096{}, cand{};
    if (!allocate(tiled4096, c, rows4096, start4096) ||
        !allocate(cand, c, rows4096, start4096))
      return 5;
    seed(tiled4096, c, rows4096, start4096);
    if (invoke_tiled(c, start4096, rows4096, tiled4096) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 6;
    std::vector<float> tiled_host(rows4096 * q);
    cudaMemcpy(tiled_host.data(), tiled4096.out,
               tiled_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    float* partial = nullptr;
    float* meta = nullptr;
    const size_t partial_n =
        qw38::cuda::fattn_stream_k_partial_values(c, rows4096);
    const size_t meta_n = qw38::cuda::fattn_stream_k_meta_values(c, rows4096);
    if (cudaMalloc(reinterpret_cast<void**>(&partial),
                   partial_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&meta), meta_n * sizeof(float)) !=
            cudaSuccess)
      return 5;
    FILE* raw = fopen("evidence/optimization/opt026-fattn-streamk/fattn-ab-raw.txt",
                      "w");
    const char* ids[3] = {"baseline", "occ2", "stream_k"};
    float means[3] = {1.0e30f, 1.0e30f, 1.0e30f};
    bool eligible[3] = {false, false, false};
    cudaEvent_t e0{}, e1{};
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < 3; ++i) {
      seed(cand, c, rows4096, start4096);
      cudaMemset(partial, 0, partial_n * sizeof(float));
      cudaMemset(meta, 0, meta_n * sizeof(float));
      std::vector<unsigned char> score_before(
          qw38::cuda::attention_chunk_score_values(c, start4096, rows4096) *
          sizeof(float));
      cudaMemcpy(score_before.data(), cand.score, score_before.size(),
                 cudaMemcpyDeviceToHost);
      AttentionCache committed{cand.ck, cand.cv}, candidate{cand.tk, cand.tv};
      cudaError_t launched = qw38::cuda::launch_attention_prepare_chunk_fattn_path(
          c, start4096, rows4096, cand.q, cand.k, cand.v, cand.qs, cand.ks,
          cand.g, committed, candidate, cand.nq, cand.nk, cand.score, cand.out,
          ids[i], i == 2 ? partial : nullptr, i == 2 ? meta : nullptr, nullptr);
      cudaError_t synced = cudaDeviceSynchronize();
      std::vector<float> host(rows4096 * q);
      if (launched == cudaSuccess && synced == cudaSuccess)
        cudaMemcpy(host.data(), cand.out, host.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
      std::vector<unsigned char> score_after(score_before.size());
      cudaMemcpy(score_after.data(), cand.score, score_after.size(),
                 cudaMemcpyDeviceToHost);
      const bool scratch = score_before == score_after;
      const bool finite = launched == cudaSuccess && synced == cudaSuccess &&
                          finite_vec(host);
      const bool env = finite && envelope_ok(host, tiled_host);
      const int occupancy =
          qw38::cuda::attention_mma_quality_occupancy_for_path(ids[i]);
      float samples[3] = {0, 0, 0};
      bool timed_ok = env && scratch && occupancy >= 1;
      for (int sample = 0; sample < 3 && timed_ok; ++sample) {
        cudaEventRecord(e0);
        timed_ok =
            qw38::cuda::launch_attention_prepare_chunk_fattn_path(
                c, start4096, rows4096, cand.q, cand.k, cand.v, cand.qs,
                cand.ks, cand.g, committed, candidate, cand.nq, cand.nk,
                cand.score, cand.out, ids[i], i == 2 ? partial : nullptr,
                i == 2 ? meta : nullptr, nullptr) == cudaSuccess;
        cudaEventRecord(e1);
        timed_ok = timed_ok && cudaEventSynchronize(e1) == cudaSuccess;
        if (timed_ok) cudaEventElapsedTime(&samples[sample], e0, e1);
      }
      const float mean =
          timed_ok ? (samples[0] + samples[1] + samples[2]) / 3.0f : 0.0f;
      eligible[i] = timed_ok && mean > 0.0f;
      means[i] = mean;
      if (raw)
        fprintf(raw,
                "fattn_ab id=%s sample=%d ms=%.9g occupancy=%d\n"
                "fattn_ab id=%s sample=%d ms=%.9g occupancy=%d\n"
                "fattn_ab id=%s sample=%d ms=%.9g occupancy=%d\n",
                ids[i], 0, samples[0], occupancy, ids[i], 1, samples[1],
                occupancy, ids[i], 2, samples[2], occupancy);
      std::printf("fattn_ab id=%s launch=%s finite=%s envelope=%s scratch=%s "
                  "occupancy=%d sample0=%.9g sample1=%.9g sample2=%.9g "
                  "mean_ms=%.9g eligible=%s\n",
                  ids[i], launched == cudaSuccess ? "ok" : "fail",
                  finite ? "true" : "false", env ? "true" : "false",
                  scratch ? "true" : "false", occupancy, samples[0], samples[1],
                  samples[2], mean, eligible[i] ? "true" : "false");
      if (raw)
        fprintf(raw,
                "fattn_ab_mean id=%s mean_ms=%.9g occupancy=%d envelope=%s "
                "scratch=%s eligible=%s\n",
                ids[i], mean, occupancy, env ? "true" : "false",
                scratch ? "true" : "false", eligible[i] ? "true" : "false");
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(partial);
    cudaFree(meta);
    release(tiled4096);
    release(cand);
    int winner_i = 0;
    bool win = false;
    if (eligible[0]) {
      float best = means[0];
      for (int i = 1; i < 3; ++i) {
        if (eligible[i] && means[i] < best) {
          best = means[i];
          winner_i = i;
        }
      }
      win = winner_i != 0 && means[winner_i] < means[0];
      if (!win) winner_i = 0;
    }
    std::printf("fattn_ab_winner id=%s win=%s baseline_ms=%.9g winner_ms=%.9g\n",
                ids[winner_i], win ? "true" : "false", means[0],
                means[winner_i]);
    if (raw) {
      fprintf(raw, "fattn_ab_winner id=%s win=%s baseline_ms=%.9g winner_ms=%.9g\n",
              ids[winner_i], win ? "true" : "false", means[0], means[winner_i]);
      fclose(raw);
    }
    if (!eligible[0]) {
      fprintf(stderr, "fattn baseline 4096 is not eligible\n");
      return 6;
    }
  }
  {
    const size_t rows4096 = 4096;
    const size_t start4096 = 0;
    B tiled4096{}, cand{};
    if (!allocate(tiled4096, c, rows4096, start4096) ||
        !allocate(cand, c, rows4096, start4096))
      return 5;
    seed(tiled4096, c, rows4096, start4096);
    if (invoke_tiled(c, start4096, rows4096, tiled4096) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 6;
    std::vector<float> tiled_host(rows4096 * q);
    cudaMemcpy(tiled_host.data(), tiled4096.out,
               tiled_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    float* partial = nullptr;
    float* meta = nullptr;
    const size_t partial_n =
        qw38::cuda::fattn_stream_k_partial_values(c, rows4096);
    const int nsm = qw38::cuda::fattn_persistent_nsm();
    const int occupancy_p =
        qw38::cuda::attention_mma_quality_occupancy_for_path("persistent");
    const int ntiles_x = static_cast<int>((rows4096 + 15) / 16);
    const int ntiles_z_gqa = 3;
    const int ntiles_dst = ntiles_x * ntiles_z_gqa * static_cast<int>(c.kv_heads);
    const int ntiles_kv = static_cast<int>((rows4096 + 31) / 32);
    const int nblocks = qw38::cuda::fattn_persistent_nblocks(
        nsm, occupancy_p, ntiles_dst, ntiles_kv);
    const size_t persist_n =
        qw38::cuda::fattn_persistent_fixup_values(nblocks);
    const size_t meta_n = qw38::cuda::fattn_stream_k_meta_values(c, rows4096);
    const size_t meta_bytes =
        (persist_n > meta_n ? persist_n : meta_n) * sizeof(float);
    if (cudaMalloc(reinterpret_cast<void**>(&partial),
                   partial_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&meta), meta_bytes) !=
            cudaSuccess)
      return 5;
    mkdir("evidence/optimization/opt027-persistent-fattn", 0755);
    FILE* raw = fopen(
        "evidence/optimization/opt027-persistent-fattn/fattn-ab-raw.txt", "w");
    const char* ids[2] = {"stream_k", "persistent"};
    float means[2] = {1.0e30f, 1.0e30f};
    bool eligible[2] = {false, false};
    cudaEvent_t e0{}, e1{};
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < 2; ++i) {
      seed(cand, c, rows4096, start4096);
      cudaMemset(partial, 0, partial_n * sizeof(float));
      cudaMemset(meta, 0, meta_bytes);
      std::vector<unsigned char> score_before(
          qw38::cuda::attention_chunk_score_values(c, start4096, rows4096) *
          sizeof(float));
      cudaMemcpy(score_before.data(), cand.score, score_before.size(),
                 cudaMemcpyDeviceToHost);
      AttentionCache committed{cand.ck, cand.cv}, candidate{cand.tk, cand.tv};
      cudaError_t launched =
          qw38::cuda::launch_attention_prepare_chunk_fattn_path(
              c, start4096, rows4096, cand.q, cand.k, cand.v, cand.qs, cand.ks,
              cand.g, committed, candidate, cand.nq, cand.nk, cand.score,
              cand.out, ids[i], partial, meta, nullptr);
      cudaError_t synced = cudaDeviceSynchronize();
      std::vector<float> host(rows4096 * q);
      if (launched == cudaSuccess && synced == cudaSuccess)
        cudaMemcpy(host.data(), cand.out, host.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
      std::vector<unsigned char> score_after(score_before.size());
      cudaMemcpy(score_after.data(), cand.score, score_after.size(),
                 cudaMemcpyDeviceToHost);
      const bool scratch = score_before == score_after;
      const bool finite = launched == cudaSuccess && synced == cudaSuccess &&
                          finite_vec(host);
      const bool env = finite && envelope_ok(host, tiled_host);
      const int occupancy =
          qw38::cuda::attention_mma_quality_occupancy_for_path(ids[i]);
      const int live_nblocks =
          i == 1 ? nblocks
                 : qw38::cuda::fattn_persistent_nblocks(
                       nsm, occupancy, ntiles_dst, ntiles_kv);
      (void)live_nblocks;
      float samples[3] = {0, 0, 0};
      bool timed_ok = env && scratch && occupancy >= 1;
      if (i == 1)
        timed_ok = timed_ok && occupancy_p >= 1 && nblocks > 0 &&
                   nblocks == qw38::cuda::fattn_persistent_nblocks(
                                  nsm, occupancy, ntiles_dst, ntiles_kv);
      for (int sample = 0; sample < 3 && timed_ok; ++sample) {
        cudaEventRecord(e0);
        timed_ok =
            qw38::cuda::launch_attention_prepare_chunk_fattn_path(
                c, start4096, rows4096, cand.q, cand.k, cand.v, cand.qs,
                cand.ks, cand.g, committed, candidate, cand.nq, cand.nk,
                cand.score, cand.out, ids[i], partial, meta, nullptr) ==
            cudaSuccess;
        cudaEventRecord(e1);
        timed_ok = timed_ok && cudaEventSynchronize(e1) == cudaSuccess;
        if (timed_ok) cudaEventElapsedTime(&samples[sample], e0, e1);
      }
      const float mean =
          timed_ok ? (samples[0] + samples[1] + samples[2]) / 3.0f : 0.0f;
      eligible[i] = timed_ok && mean > 0.0f;
      means[i] = mean;
      if (raw)
        fprintf(raw,
                "fattn_ab id=%s sample=%d ms=%.9g occupancy=%d nsm=%d "
                "nblocks=%d\n"
                "fattn_ab id=%s sample=%d ms=%.9g occupancy=%d\n"
                "fattn_ab id=%s sample=%d ms=%.9g occupancy=%d\n",
                ids[i], 0, samples[0], occupancy, nsm, nblocks, ids[i], 1,
                samples[1], occupancy, ids[i], 2, samples[2], occupancy);
      std::printf("opt027_fattn_ab id=%s launch=%s finite=%s envelope=%s "
                  "scratch=%s occupancy=%d nsm=%d nblocks=%d sample0=%.9g "
                  "sample1=%.9g sample2=%.9g mean_ms=%.9g eligible=%s\n",
                  ids[i], launched == cudaSuccess ? "ok" : "fail",
                  finite ? "true" : "false", env ? "true" : "false",
                  scratch ? "true" : "false", occupancy, nsm, nblocks,
                  samples[0], samples[1], samples[2], mean,
                  eligible[i] ? "true" : "false");
      if (raw)
        fprintf(raw,
                "fattn_ab_mean id=%s mean_ms=%.9g occupancy=%d envelope=%s "
                "scratch=%s eligible=%s nsm=%d nblocks=%d\n",
                ids[i], mean, occupancy, env ? "true" : "false",
                scratch ? "true" : "false", eligible[i] ? "true" : "false",
                nsm, nblocks);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(partial);
    cudaFree(meta);
    release(tiled4096);
    release(cand);
    int winner_i = 0;
    bool win = false;
    if (eligible[0] && eligible[1] && means[1] < means[0]) {
      winner_i = 1;
      win = true;
    }
    std::printf("opt027_fattn_ab_winner id=%s win=%s stream_k_ms=%.9g "
                "winner_ms=%.9g\n",
                ids[winner_i], win ? "true" : "false", means[0],
                means[winner_i]);
    if (raw) {
      fprintf(raw,
              "fattn_ab_winner id=%s win=%s stream_k_ms=%.9g winner_ms=%.9g\n",
              ids[winner_i], win ? "true" : "false", means[0],
              means[winner_i]);
      fclose(raw);
    }
    if (!eligible[0]) {
      fprintf(stderr, "opt027 stream_k 4096 is not eligible\n");
      return 6;
    }
  }
  {
    const size_t rows4096 = 4096;
    const size_t start4096 = 0;
    B tiled4096{}, cand{};
    if (!allocate(tiled4096, c, rows4096, start4096) ||
        !allocate(cand, c, rows4096, start4096))
      return 5;
    seed(tiled4096, c, rows4096, start4096);
    if (invoke_tiled(c, start4096, rows4096, tiled4096) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 6;
    std::vector<float> tiled_host(rows4096 * q);
    cudaMemcpy(tiled_host.data(), tiled4096.out,
               tiled_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    float* partial = nullptr;
    float* meta = nullptr;
    const size_t partial_n =
        qw38::cuda::fattn_stream_k_partial_values(c, rows4096);
    const size_t meta_n = qw38::cuda::fattn_stream_k_meta_values(c, rows4096);
    if (cudaMalloc(reinterpret_cast<void**>(&partial),
                   partial_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&meta), meta_n * sizeof(float)) !=
            cudaSuccess)
      return 5;
    seed(cand, c, rows4096, start4096);
    std::vector<__nv_bfloat16> committed_key_before(
        qw38::cuda::attention_cache_values(c));
    std::vector<__nv_bfloat16> committed_value_before(committed_key_before.size());
    cudaMemcpy(committed_key_before.data(), cand.ck,
               committed_key_before.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(committed_value_before.data(), cand.cv,
               committed_value_before.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    std::vector<unsigned char> score_before(
        qw38::cuda::attention_chunk_score_values(c, start4096, rows4096) *
        sizeof(float));
    cudaMemcpy(score_before.data(), cand.score, score_before.size(),
               cudaMemcpyDeviceToHost);
    AttentionCache committed{cand.ck, cand.cv}, candidate{cand.tk, cand.tv};
    cudaError_t launched = qw38::cuda::launch_attention_prepare_chunk_stream_k_vkq(
        c, start4096, rows4096, cand.q, cand.k, cand.v, cand.qs, cand.ks,
        cand.g, committed, candidate, cand.nq, cand.nk, cand.score, cand.out,
        partial, meta, "registers", nullptr);
    cudaError_t synced = cudaDeviceSynchronize();
    std::vector<float> host(rows4096 * q);
    if (launched == cudaSuccess && synced == cudaSuccess)
      cudaMemcpy(host.data(), cand.out, host.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
    std::vector<unsigned char> score_after(score_before.size());
    cudaMemcpy(score_after.data(), cand.score, score_after.size(),
               cudaMemcpyDeviceToHost);
    std::vector<__nv_bfloat16> committed_key_after(committed_key_before.size());
    std::vector<__nv_bfloat16> committed_value_after(committed_value_before.size());
    cudaMemcpy(committed_key_after.data(), cand.ck,
               committed_key_after.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(committed_value_after.data(), cand.cv,
               committed_value_after.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    const bool scratch = score_before == score_after;
    const bool committed_unchanged =
        memcmp(committed_key_before.data(), committed_key_after.data(),
               committed_key_before.size() * sizeof(__nv_bfloat16)) == 0 &&
        memcmp(committed_value_before.data(), committed_value_after.data(),
               committed_value_before.size() * sizeof(__nv_bfloat16)) == 0;
    const bool finite = launched == cudaSuccess && synced == cudaSuccess &&
                        finite_vec(host);
    const bool env = finite && envelope_ok(host, tiled_host);
    const int occupancy = qw38::cuda::fattn_register_vkq_occupancy();
    std::printf("opt033_register_vkq launch=%s finite=%s envelope=%s "
                "scratch=%s committed=%s occupancy=%d max_abs=%.9g rms=%.9g\n",
                launched == cudaSuccess ? "ok" : "fail",
                finite ? "true" : "false", env ? "true" : "false",
                scratch ? "true" : "false",
                committed_unchanged ? "true" : "false", occupancy,
                finite ? maxabs(host, tiled_host) : 0.0, finite ? rms(host, tiled_host) : 0.0);
    cudaFree(partial);
    cudaFree(meta);
    release(tiled4096);
    release(cand);
    if (!env || !scratch || !committed_unchanged || occupancy < 1) {
      fprintf(stderr, "opt033 register vkq 4096 failed envelope or isolation\n");
      return 6;
    }
  }
  {
    const size_t rows4096 = 4096;
    const size_t start4096 = 0;
    B tiled4096{}, cand{};
    if (!allocate(tiled4096, c, rows4096, start4096) ||
        !allocate(cand, c, rows4096, start4096))
      return 5;
    seed(tiled4096, c, rows4096, start4096);
    if (invoke_tiled(c, start4096, rows4096, tiled4096) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 6;
    std::vector<float> tiled_host(rows4096 * q);
    cudaMemcpy(tiled_host.data(), tiled4096.out,
               tiled_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    float* partial = nullptr;
    float* meta = nullptr;
    const size_t partial_n =
        qw38::cuda::fattn_stream_k_partial_values(c, rows4096);
    const size_t meta_n = qw38::cuda::fattn_stream_k_meta_values(c, rows4096);
    if (cudaMalloc(reinterpret_cast<void**>(&partial),
                   partial_n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&meta), meta_n * sizeof(float)) !=
            cudaSuccess)
      return 5;
    seed(cand, c, rows4096, start4096);
    std::vector<__nv_bfloat16> committed_key_before(
        qw38::cuda::attention_cache_values(c));
    std::vector<__nv_bfloat16> committed_value_before(committed_key_before.size());
    cudaMemcpy(committed_key_before.data(), cand.ck,
               committed_key_before.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(committed_value_before.data(), cand.cv,
               committed_value_before.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    std::vector<unsigned char> score_before(
        qw38::cuda::attention_chunk_score_values(c, start4096, rows4096) *
        sizeof(float));
    cudaMemcpy(score_before.data(), cand.score, score_before.size(),
               cudaMemcpyDeviceToHost);
    AttentionCache committed{cand.ck, cand.cv}, candidate{cand.tk, cand.tv};
    cudaError_t launched = qw38::cuda::launch_attention_prepare_chunk_stream_k_pv(
        c, start4096, rows4096, cand.q, cand.k, cand.v, cand.qs, cand.ks,
        cand.g, committed, candidate, cand.nq, cand.nk, cand.score, cand.out,
        partial, meta, "mma", nullptr);
    cudaError_t synced = cudaDeviceSynchronize();
    std::vector<float> host(rows4096 * q);
    if (launched == cudaSuccess && synced == cudaSuccess)
      cudaMemcpy(host.data(), cand.out, host.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
    std::vector<unsigned char> score_after(score_before.size());
    cudaMemcpy(score_after.data(), cand.score, score_after.size(),
               cudaMemcpyDeviceToHost);
    std::vector<__nv_bfloat16> committed_key_after(committed_key_before.size());
    std::vector<__nv_bfloat16> committed_value_after(committed_value_before.size());
    cudaMemcpy(committed_key_after.data(), cand.ck,
               committed_key_after.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(committed_value_after.data(), cand.cv,
               committed_value_after.size() * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    const bool scratch = score_before == score_after;
    const bool committed_unchanged =
        memcmp(committed_key_before.data(), committed_key_after.data(),
               committed_key_before.size() * sizeof(__nv_bfloat16)) == 0 &&
        memcmp(committed_value_before.data(), committed_value_after.data(),
               committed_value_before.size() * sizeof(__nv_bfloat16)) == 0;
    const bool finite = launched == cudaSuccess && synced == cudaSuccess &&
                        finite_vec(host);
    const bool env = finite && envelope_ok(host, tiled_host);
    const int occupancy = qw38::cuda::fattn_pv_mma_occupancy();
    std::printf("opt035_pv_mma launch=%s finite=%s envelope=%s "
                "scratch=%s committed=%s occupancy=%d max_abs=%.9g rms=%.9g\n",
                launched == cudaSuccess ? "ok" : "fail",
                finite ? "true" : "false", env ? "true" : "false",
                scratch ? "true" : "false",
                committed_unchanged ? "true" : "false", occupancy,
                finite ? maxabs(host, tiled_host) : 0.0, finite ? rms(host, tiled_host) : 0.0);
    cudaFree(partial);
    cudaFree(meta);
    release(tiled4096);
    release(cand);
    if (launched != cudaSuccess || synced != cudaSuccess || !scratch ||
        !committed_unchanged) {
      fprintf(stderr, "opt035 pv mma 4096 failed launch or isolation\n");
      return 6;
    }
    if (qw38::cuda::fattn_uses_pv_mma() &&
        (!env || occupancy < 1 || !finite)) {
      fprintf(stderr, "opt035 production mma 4096 failed envelope or occupancy\n");
      return 6;
    }
  }
  B graph{};if(!allocate(graph,c,64,start))return 5;seed(graph,c,64,start);int p1=capture(c,start,1,graph,false),p3=capture(c,start,3,graph,false),p9=capture(c,start,9,graph,false),p64=capture(c,start,64,graph,false),r1=capture(c,start,1,graph,true),r3=capture(c,start,3,graph,true),r9=capture(c,start,9,graph,true),r64=capture(c,start,64,graph,true);release(graph);bool graphs=p1==2&&p3==2&&p9==2&&p64==2&&r1==3&&r3==9&&r9==27&&r64==192;
  bool semantic=finite&&candidate_exact&&chunk_output&&chunk_candidate&&cache_unchanged&&frontier_unchanged&&commit_exact&&commit_frontier&&future_excluded&&later_excluded&&scratch&&zero_rejected&&overflow_rejected&&alias_rejected&&last_position&&normalized_equal&&graphs&&ma<=5e-5f&&rr<=5e-6f&&co>=.999424f&&ma3<=5e-5f&&rr3<=5e-6f&&co3>=.999424f;if(!semantic){fprintf(stderr,"semantic failure finite=%d candidate=%d repeated_output=%d repeated_candidate=%d cache=%d frontier=%d commit=%d commit_frontier=%d future=%d later=%d scratch=%d zero=%d overflow=%d alias=%d last=%d normalized=%d graphs=%d metrics9=%d metrics3=%d\n",finite,candidate_exact,chunk_output,chunk_candidate,cache_unchanged,frontier_unchanged,commit_exact,commit_frontier,future_excluded,later_excluded,scratch,zero_rejected,overflow_rejected,alias_rejected,last_position,normalized_equal,graphs,ma<=5e-5f&&rr<=5e-6f&&co>=.999424f,ma3<=5e-5f&&rr3<=5e-6f&&co3>=.999424f);return 6;}
  struct Scale{size_t prefix;std::vector<float>tiled,reference;double tm,rm;};std::vector<Scale> scales;
  for(size_t prefix:{size_t(2048),size_t(8192),size_t(32768)}){rows=64;B x{},y{};if(!allocate(x,c,rows,prefix)||!allocate(y,c,rows,prefix))return 5;seed(x,c,rows,prefix);seed(y,c,rows,prefix);for(int i=0;i<3;++i)invoke(c,prefix,rows,x,false,nullptr);cudaDeviceSynchronize();cudaEvent_t e0{},e1{};cudaEventCreate(&e0);cudaEventCreate(&e1);Scale z{prefix,std::vector<float>(30),std::vector<float>(3),0,0};for(float&v:z.tiled){cudaEventRecord(e0);if(invoke(c,prefix,rows,x,false,nullptr)!=cudaSuccess)return 7;cudaEventRecord(e1);if(cudaEventSynchronize(e1)!=cudaSuccess)return 7;cudaEventElapsedTime(&v,e0,e1);}invoke(c,prefix,rows,y,true,nullptr);cudaDeviceSynchronize();for(float&v:z.reference){cudaEventRecord(e0);if(invoke(c,prefix,rows,y,true,nullptr)!=cudaSuccess)return 7;cudaEventRecord(e1);if(cudaEventSynchronize(e1)!=cudaSuccess)return 7;cudaEventElapsedTime(&v,e0,e1);}for(float v:z.tiled)z.tm+=v;for(float v:z.reference)z.rm+=v;z.tm/=30;z.rm/=3;cudaEventDestroy(e0);cudaEventDestroy(e1);release(x);release(y);if(!(z.tm>0&&z.rm>z.tm))return 8;scales.push_back(z);}
  cudaDeviceProp prop{};int dev=0,driver=0,runtime=0;cudaGetDevice(&dev);cudaGetDeviceProperties(&prop,dev);cudaDriverGetVersion(&driver);cudaRuntimeGetVersion(&runtime);std::time_t now=std::time(nullptr);char utc[32]{};std::tm tm{};gmtime_r(&now,&tm);std::strftime(utc,sizeof(utc),"%Y-%m-%dT%H:%M:%SZ",&tm);
  printf("QW38_TILED_ATTENTION_RESULT={\"schema_version\":2,\"task\":\"OPT-005\",\"status\":\"measured\",\"device\":\"%s\",\"compute_capability\":\"%d.%d\",\"driver\":\"%d.%d\",\"runtime\":\"%d.%d\",\"toolkit\":\"CUDA 13.0.2\",\"pinned_image\":\"qw38-cuda:13.0.2\",\"measurement_utc\":\"%s\",\"warmups\":{\"tiled\":3,\"reference\":1},\"proof_limits\":{\"bf16_max_abs\":0.00005,\"bf16_rms\":0.000005,\"oracle_max_abs\":0.051,\"oracle_rms\":0.0016,\"oracle_cosine\":0.999424},\"semantic\":{\"predicates\":{",prop.name,prop.major,prop.minor,driver/1000,(driver%1000)/10,runtime/1000,(runtime%1000)/10,utc);
  const char* names[]={"finite_output","candidate_bf16_exact","chunk_repeated_output_exact","chunk_repeated_candidate_exact","prepare_cache_unchanged","prepare_frontier_unchanged","commit_cache_exact","commit_frontier_exact","future_committed_excluded","later_candidate_excluded","scratch_unchanged","zero_count_rejected","capacity_overflow_rejected","candidate_alias_rejected","last_position_executed","normalized_qk_one_row_equal","graphs_executed"};for(int i=0;i<17;++i)printf("\"%s\":true%s",names[i],i==16?"":",");printf("},\"metrics\":{\"3\":{\"max_abs\":%.9g,\"rms\":%.9g,\"cosine\":%.9g},\"9\":{\"max_abs\":%.9g,\"rms\":%.9g,\"cosine\":%.9g}},\"production_kernel_nodes\":{\"1\":%d,\"3\":%d,\"9\":%d,\"64\":%d},\"reference_kernel_nodes\":{\"1\":%d,\"3\":%d,\"9\":%d,\"64\":%d}},\"scaling\":[",ma3,rr3,co3,ma,rr,co,p1,p3,p9,p64,r1,r3,r9,r64);
  for(size_t j=0;j<scales.size();++j){const auto&z=scales[j];printf("{\"prefix\":%zu,\"rows\":64,\"tiled_samples\":[",z.prefix);for(size_t i=0;i<z.tiled.size();++i)printf("%.9g%s",z.tiled[i],i+1==z.tiled.size()?"":",");printf("],\"reference_samples\":[");for(size_t i=0;i<z.reference.size();++i)printf("%.9g%s",z.reference[i],i+1==z.reference.size()?"":",");printf("],\"mean_ms_tiled\":%.9g,\"mean_ms_reference\":%.9g,\"speedup\":%.9g}%s",z.tm,z.rm,z.rm/z.tm,j+1==scales.size()?"":",");}
  printf("],\"claim\":\"component-only production-shape tiled attention timing; excludes projections, scheduler, and end-to-end recovery\"}\n");return 0;}
