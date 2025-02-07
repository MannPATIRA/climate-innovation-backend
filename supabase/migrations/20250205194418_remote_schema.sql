alter table "public"."papers" drop column "abstract";

alter table "public"."processor_progress" add column "current_cursor" character varying default '*'::character varying;

alter table "public"."processor_progress" add column "cursor" character varying default '*'::character varying;


