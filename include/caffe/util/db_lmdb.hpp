#ifdef USE_LMDB
#ifndef CAFFE_UTIL_DB_LMDB_HPP
#define CAFFE_UTIL_DB_LMDB_HPP

#include <string>
#include <vector>

#include "lmdb.h"

#include "caffe/util/db.hpp"

namespace caffe { 
namespace db {
//数据库确定工具函数
inline void MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}
//数据查询指针，指向创建的数据库连接
class LMDBCursor : public Cursor {
 public:
	 //构造函数
  explicit LMDBCursor(MDB_txn* mdb_txn, //上下文 
					  MDB_cursor* mdb_cursor //操作指针
					 )
    : mdb_txn_(mdb_txn), mdb_cursor_(mdb_cursor), valid_(false) {
    SeekToFirst();
  }
  virtual ~LMDBCursor() {
	  //关闭查询连接
    mdb_cursor_close(mdb_cursor_);
	//关闭上下文
    mdb_txn_abort(mdb_txn_);
  }
  //查找第一个
  virtual void SeekToFirst() { Seek(MDB_FIRST); }
  virtual void Next() { Seek(MDB_NEXT); }
  //key键
  virtual string key() {
    return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  }
  //获取查询结果
  virtual string value() {
    return string(static_cast<const char*>(mdb_value_.mv_data),
        mdb_value_.mv_size);
  }
  virtual bool valid() { return valid_; }

 private:
	 //寻找关键字和值
  void Seek(MDB_cursor_op op) {
    int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      valid_ = false;
    } else {
      MDB_CHECK(mdb_status);
      valid_ = true;
    }
  }
	//mdb上下文
  MDB_txn* mdb_txn_;
  //mdb数据指针
  MDB_cursor* mdb_cursor_;
  //关键字和值
  MDB_val mdb_key_, mdb_value_;
	//是否存在
  bool valid_;
};
//lmdb事物
class LMDBTransaction : public Transaction {
 public:
  explicit LMDBTransaction(MDB_env* mdb_env)
    : mdb_env_(mdb_env) { }
	//插入关键字
  virtual void Put(const string& key, const string& value);
  //提交需改函数
  virtual void Commit();

 private:
  MDB_env* mdb_env_;
  //键值对
  vector<string> keys, values;
	//数据库扩容
  void DoubleMapSize();

  DISABLE_COPY_AND_ASSIGN(LMDBTransaction);
};
//LMDB类,主要功能是创建连接，帮助生成 LMDBCursor和LMDBTransaction
class LMDB : public DB {
 public:
  LMDB() : mdb_env_(NULL) { }
  virtual ~LMDB() { Close(); }
  //打开数据库
  virtual void Open(const string& source, Mode mode);
  //关闭数据库
  virtual void Close() {
    if (mdb_env_ != NULL) {
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_env_close(mdb_env_);
      mdb_env_ = NULL;
    }
  }
  //查询指针
  virtual LMDBCursor* NewCursor();
  //事物指针
  virtual LMDBTransaction* NewTransaction();

 private:
	 //环境上下文
  MDB_env* mdb_env_;
  //索引
  MDB_dbi mdb_dbi_;
};
//这个类主要是对lmdb的一个简单封装
}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LMDB_HPP
#endif  // USE_LMDB
