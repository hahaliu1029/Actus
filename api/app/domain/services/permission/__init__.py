"""Permission engine subpackage.

Type-safe facade over the existing react_graph._run_policy_chain (line 260
in 2026-05-14 working tree). All approval / deny / ask decisions for tool
calls flow through PermissionEngine; ApprovalStateWriter (R5 CS4) is called
only from within DefaultPermissionEngine.

INV-1b (CI gate): writer.write / write_audit_only / delete_grant outside
this package or its DI factory fails the build.
"""
