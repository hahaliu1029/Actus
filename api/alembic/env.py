from logging.config import fileConfig

from alembic import context
from app.infrastructure.models import Base  # 导入模型的Base以获取MetaData
from sqlalchemy import engine_from_config, pool

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Skip alembic's logging config entirely. The application's
# ``app.infrastructure.logging.setup_logging`` (called at module import
# time of ``app.main``) already installs a root StreamHandler at INFO
# level pointing at sys.stdout. If we call ``fileConfig`` here, alembic
# would re-configure the root logger using ``[logger_root]`` from
# ``alembic.ini`` (level=WARNING, handler→sys.stderr), which would
# (a) demote root to WARNING — silencing main.py's own INFO logs like
# ``数据库迁移完成`` and ``Postgres 客户端初始化完成``,
# (b) optionally disable every ``app.*`` logger not listed in alembic.ini.
# Skipping fileConfig leaves the root logger as setup_logging configured
# it; alembic's own loggers (``logging.getLogger("alembic")``,
# ``logging.getLogger("sqlalchemy.engine")``) inherit from root and still
# emit normally.
#
# Note: this means ``alembic.ini``'s [logger_*] / [handler_*] /
# [formatter_*] sections are now effectively dead config inside the
# Actus application. Standalone ``alembic upgrade head`` from the CLI
# still uses them because the CLI doesn't import ``app.main`` first.
_ = fileConfig  # keep the import live for lints / future fallback use

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
