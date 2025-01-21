alter table "public"."chat_messages" alter column "order" drop default;

alter table "public"."chat_messages" alter column "order" set not null;


