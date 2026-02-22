from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

# 定义命名约定
naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# 将约定传递给 MetaData
Base = declarative_base(metadata=MetaData(naming_convention=naming_convention))
